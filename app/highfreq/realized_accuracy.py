"""Rolling realized-accuracy of the paper trader, computed from
``paper_trades``.

What this measures
==================

The walk-forward CV in ``app.highfreq.trainer`` answers "is the model
better than chance on **historical** data?" — replay value from the
trainer's perspective. This module answers a different and harder
question:

> *Is the trader, **right now in production**, beating chance on the
> trades it actually closed?*

Production realized accuracy is what reviewers care about. CV results
that don't carry into deployment usually mean (a) train-serve feature
drift, (b) market regime shift since the trainer last fit, or (c) the
trader's threshold/risk-cap logic strips signal at deployment. Watching
this number drift is how we'd notice — long before P&L tells us — that
something in the live stack regressed.

What goes into the sample
-------------------------

Each closed paper trade has ``side`` (``long`` / ``short``) — that's the
trader's *predicted* direction — and ``entry_price`` / ``exit_price``,
from which we derive the **realized** direction. A trade is *correct*
when:

* ``side="long"`` and ``exit_price > entry_price``, OR
* ``side="short"`` and ``exit_price < entry_price``.

``halt_close`` exits are **excluded** from the sample. Risk caps fire
the close immediately at the current bar's microprice — the trade
never got to run its full minute, so its realized return is not a
fair test of the model's directional call. ``time_stop`` exits (the
clean 1-minute close) are the only meaningful sample.

Tied bars (``exit_price == entry_price``) get classified as **not
correct**: the directional call was wrong in the sense that no
profitable move materialised. This conservative choice avoids
"counting ties as half-correct" sleight-of-hand when reporting the
number on slides.

Why this lives in its own module
--------------------------------

* Pure-logic code (``compute_rolling_accuracy_from_rows``) is unit-
  tested without a DB — the caller drives whatever rows it wants.
* The DB layer (``fetch_rolling_accuracy``) is async-only because the
  paper-trader runner is async (asyncpg pool). Making the math pure
  means we can also import it from the **sync** FastAPI endpoint in
  ``app.highfreq.web`` (where we use SQLAlchemy sync sessions) by
  passing rows in directly.
* Splitting "compute" from "fetch" stops the tests from creeping into
  asyncio fixtures, which keeps the test suite fast.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Iterable, Optional, Sequence


# Defaults: 50 captures recent trader skill, 100 averages over a half-day
# at typical paper-trader cadence (signals fire every minute → ~24/h, so
# 100 ≈ 4 hours of trading).  Both are exposed as labels on the
# Prometheus gauge so a single dashboard can plot fast vs slow.
DEFAULT_WINDOW_SIZES: tuple[int, ...] = (50, 100)


@dataclass(frozen=True)
class TradeRow:
    """Minimal row shape for accuracy math.

    Decoupled from the full ``paper_trades`` row because (a) we don't
    need fee/pnl fields here and (b) keeping it small keeps the tests
    constructor-free of irrelevant fields.
    """
    symbol: str
    side: str                # "long" | "short"
    entry_price: float
    exit_price: float
    exit_reason: str         # "time_stop" | "halt_close"
    exit_ts: datetime
    entry_prob_up: float


@dataclass(frozen=True)
class RealizedAccuracy:
    """JSON-friendly snapshot of the rolling-accuracy computation.

    Mirrors the shape returned by the API endpoint and the gauge label
    space. ``None`` for ``accuracy`` etc. when the window has zero
    eligible trades — the UI must render a "—" rather than a confusing
    ``0.00``."""
    symbol: str
    window: int
    n_trades_total: int       # rows considered, including halt_close
    n_eligible: int           # rows after filtering halt_close
    n_correct: int
    accuracy: float | None    # n_correct / n_eligible; None if n_eligible == 0
    avg_predicted_proba_up: float | None
    earliest_exit_ts: str | None
    latest_exit_ts: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_rolling_accuracy_from_rows(
    rows: Sequence[TradeRow],
    *,
    symbol: str,
    window: int,
) -> RealizedAccuracy:
    """Pure: compute the rolling-accuracy snapshot from already-fetched
    rows.

    The caller is responsible for ordering ``rows`` exit-time DESC and
    for slicing to the rolling window. We do not re-sort here — that
    would (a) require an unnecessary DataFrame import, (b) hide the
    intent that the SQL ORDER BY does the heavy lifting.

    We do still iterate to filter ``halt_close`` so the test side can
    pass a deliberately "wrong" mix and observe the filter at work.
    """
    n_total = len(rows)
    eligible = [r for r in rows if r.exit_reason == "time_stop"]
    n_eligible = len(eligible)

    if n_eligible == 0:
        return RealizedAccuracy(
            symbol=symbol,
            window=window,
            n_trades_total=n_total,
            n_eligible=0,
            n_correct=0,
            accuracy=None,
            avg_predicted_proba_up=None,
            earliest_exit_ts=None,
            latest_exit_ts=None,
        )

    n_correct = sum(1 for r in eligible if _is_directional_hit(r))
    avg_proba = sum(r.entry_prob_up for r in eligible) / n_eligible
    # ``rows`` is exit-time DESC by SQL contract. earliest/latest:
    earliest = min(r.exit_ts for r in eligible)
    latest = max(r.exit_ts for r in eligible)

    return RealizedAccuracy(
        symbol=symbol,
        window=window,
        n_trades_total=n_total,
        n_eligible=n_eligible,
        n_correct=n_correct,
        accuracy=n_correct / n_eligible,
        avg_predicted_proba_up=avg_proba,
        earliest_exit_ts=earliest.isoformat() if earliest else None,
        latest_exit_ts=latest.isoformat() if latest else None,
    )


def _is_directional_hit(r: TradeRow) -> bool:
    """``True`` iff side and realized direction agree.

    Ties (exit == entry) are counted as **misses** — see module docstring
    rationale.
    """
    if r.side == "long":
        return r.exit_price > r.entry_price
    if r.side == "short":
        return r.exit_price < r.entry_price
    # Defensive: an unknown side from the DB (CHECK constraint should
    # prevent this) — be loud rather than silently reporting 0% accuracy.
    raise ValueError(f"unknown side: {r.side!r}")


# ──────────────────────────────────────────────────────────────────────
# DB layer (async — used by the paper-trader runner)
# ──────────────────────────────────────────────────────────────────────


async def fetch_rolling_accuracy(
    pool: Any,  # asyncpg.Pool — typed as Any to keep import optional
    *,
    symbol: str,
    window: int,
) -> RealizedAccuracy:
    """Fetch the last ``window`` paper trades for ``symbol`` and compute
    the rolling-accuracy snapshot.

    ``halt_close`` rows are pulled too (they go into ``n_trades_total``)
    but excluded from the accuracy denominator inside
    :func:`compute_rolling_accuracy_from_rows`.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    sql = (
        "SELECT symbol, side, entry_price, exit_price, exit_reason, "
        "       exit_ts, entry_prob_up "
        "  FROM paper_trades "
        " WHERE symbol = $1 "
        " ORDER BY exit_ts DESC "
        " LIMIT $2"
    )
    async with pool.acquire() as conn:
        records = await conn.fetch(sql, symbol, window)

    rows = [
        TradeRow(
            symbol=r["symbol"],
            side=r["side"],
            entry_price=float(r["entry_price"]),
            exit_price=float(r["exit_price"]),
            exit_reason=r["exit_reason"],
            exit_ts=r["exit_ts"],
            entry_prob_up=float(r["entry_prob_up"]),
        )
        for r in records
    ]
    return compute_rolling_accuracy_from_rows(rows, symbol=symbol, window=window)


# ──────────────────────────────────────────────────────────────────────
# DB layer (sync — used by the FastAPI endpoint via SQLAlchemy)
# ──────────────────────────────────────────────────────────────────────


def fetch_rolling_accuracy_sync(
    db_session: Any,  # sqlalchemy.orm.Session — Any to avoid heavy import
    *,
    symbol: str,
    window: int,
) -> RealizedAccuracy:
    """Sync variant for the FastAPI endpoint.

    The web layer in ``app.highfreq.web`` uses SQLAlchemy sync Sessions
    (chosen there to keep the FastAPI endpoint deps minimal). This
    helper lets that endpoint reuse the exact same math + filtering,
    so the rolling-accuracy number reported in the JSON API and on the
    Prometheus gauge cannot drift apart.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    from sqlalchemy import text  # local import — keeps non-web modules light
    sql = text(
        "SELECT symbol, side, entry_price, exit_price, exit_reason, "
        "       exit_ts, entry_prob_up "
        "  FROM paper_trades "
        " WHERE symbol = :symbol "
        " ORDER BY exit_ts DESC "
        " LIMIT :window"
    )
    res = db_session.execute(sql, {"symbol": symbol, "window": window})
    rows = [
        TradeRow(
            symbol=r.symbol,
            side=r.side,
            entry_price=float(r.entry_price),
            exit_price=float(r.exit_price),
            exit_reason=r.exit_reason,
            exit_ts=r.exit_ts,
            entry_prob_up=float(r.entry_prob_up),
        )
        for r in res
    ]
    return compute_rolling_accuracy_from_rows(rows, symbol=symbol, window=window)
