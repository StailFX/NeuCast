"""FastAPI router for the high-frequency `/highfreq` page (Phase A.6).

Read-only UI surface over the data the L2 ingestor writes into Postgres:

* ``GET /highfreq``                 — HTML page (live microprice + countdown)
* ``GET /api/highfreq/status``      — JSON poll endpoint (2-second cadence)
* ``GET /api/highfreq/health``      — quick liveness probe (200 / 503)

Design notes
------------

The L2 ingestor is a long-running asyncio service writing rows via
``asyncpg`` (see :mod:`app.highfreq.aggregator`). The FastAPI process
intentionally does **not** share that connection pool — it issues short
synchronous read queries on the main SQLAlchemy engine, matching the
pattern used by the rest of ``app.main``.

This means the UI is a "thin observer" of the same Postgres tables and
will keep working even if the ingest service is restarting.

Phase A scope is sim-only — the page must visibly reflect ADR-005:
**no live trading, paper P&L only**. See ``docs/highfreq/architecture.md``.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError, OperationalError
from sqlalchemy.orm import Session

from app.highfreq.feature_pipeline import build_latest_feature_row
from app.highfreq.paper_trader import PaperTraderConfig, RiskCaps
from app.highfreq.predictor import LivePredictor, get_predictor

# NOTE: ``app.db`` reads ``DATABASE_URL`` at import time, so we **defer**
# importing ``SessionLocal`` to the request-scoped dependency below. This
# keeps ``app.highfreq.web`` importable in test contexts without env config
# (matches the pattern used by ``app.highfreq.trainer.run_training``).

logger = logging.getLogger(__name__)


# ── Constants tied to architecture decisions ──────────────────────────────
# The walk-forward trainer needs ≥(initial_train_minutes + test_fold_minutes)
# minutes of data before it can produce its first fold report. With current
# defaults (initial=5, test=60) the minimum is 65 minutes; we surface this
# to users as a countdown so they know when training becomes feasible.
MIN_MINUTES_FOR_TRAINING: int = 65

DEFAULT_SYMBOL: str = os.getenv("HIGHFREQ_DEFAULT_SYMBOL", "BTCUSDT")

# A row is considered "fresh" if its ts is within this many seconds of now.
# The aggregator emits at 1-second cadence, so anything older than ~10s
# is a clear sign the ingest pipeline has stalled.
LIVENESS_THRESHOLD_SEC: int = 15


# ── Pure-logic dataclasses (testable without a DB) ────────────────────────


@dataclass(frozen=True)
class HighfreqSnapshot:
    """Most recent 1-second row joined with derived freshness signals."""

    ts: Optional[datetime]
    symbol: str
    microprice: Optional[float]
    ofi: Optional[float]
    depth_imb: Optional[float]
    spread_bps: Optional[float]
    age_seconds: Optional[float]


@dataclass(frozen=True)
class HighfreqStatus:
    """Full payload returned by ``GET /api/highfreq/status``.

    All fields are JSON-friendly (no NaN / Inf — see :func:`_scrub`).
    """

    symbol: str
    snapshot: Optional[HighfreqSnapshot]
    minutes_accumulated: int
    minutes_required: int
    minutes_remaining: int
    progress_pct: float                  # 0.0 .. 100.0
    is_live: bool                         # latest row is fresh
    has_enough_data: bool                 # ≥ MIN_MINUTES_FOR_TRAINING
    rows_per_second_estimate: Optional[float]  # last 60s
    server_time: datetime

    def to_dict(self) -> dict[str, Any]:
        return _scrub(asdict(self))


# ── JSON sanitisation ─────────────────────────────────────────────────────


def _scrub(value: Any) -> Any:
    """Recursively replace NaN / Inf with ``None`` for RFC-7159 compliance.

    Same pattern used in ``app.highfreq.trainer.TrainingReport.to_json``.
    """
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


# ── Pure-logic helpers (unit-tested without spinning up FastAPI) ──────────


def compute_progress_fields(
    minutes_accumulated: int,
    minutes_required: int = MIN_MINUTES_FOR_TRAINING,
) -> tuple[int, float, bool]:
    """Derive (remaining, progress_pct, has_enough_data) from raw count.

    ``minutes_accumulated`` may exceed the requirement; ``progress_pct`` is
    clamped to ``100.0`` and ``minutes_remaining`` floors at zero.
    """
    accumulated = max(0, int(minutes_accumulated))
    required = max(1, int(minutes_required))
    remaining = max(0, required - accumulated)
    progress = min(100.0, 100.0 * accumulated / required)
    has_enough = accumulated >= required
    return remaining, progress, has_enough


def compute_age_seconds(ts: Optional[datetime], now: datetime) -> Optional[float]:
    """Seconds elapsed between ``ts`` and ``now`` (UTC-aware), or ``None``."""
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    delta = (now - ts).total_seconds()
    # Clamp tiny clock-skew negatives — they confuse the UI but are not bugs.
    return max(0.0, float(delta))


def assemble_status(
    *,
    symbol: str,
    last_row: Optional[dict[str, Any]],
    minutes_accumulated: int,
    rows_last_60s: Optional[int],
    now: datetime,
) -> HighfreqStatus:
    """Pure function that turns raw DB results into the status payload.

    Split out so unit tests can exercise the field-derivation logic without
    a live database. The router :func:`get_status` composes the three
    queries and calls this.
    """
    if last_row is not None:
        ts = last_row.get("ts")
        age = compute_age_seconds(ts, now)
        snapshot = HighfreqSnapshot(
            ts=ts,
            symbol=str(last_row.get("symbol", symbol)),
            microprice=_to_float(last_row.get("microprice")),
            ofi=_to_float(last_row.get("ofi")),
            depth_imb=_to_float(last_row.get("depth_imb")),
            spread_bps=_to_float(last_row.get("spread_bps")),
            age_seconds=age,
        )
        is_live = age is not None and age <= LIVENESS_THRESHOLD_SEC
    else:
        snapshot = None
        is_live = False

    remaining, progress, has_enough = compute_progress_fields(minutes_accumulated)

    rate: Optional[float]
    if rows_last_60s is None:
        rate = None
    else:
        rate = round(float(rows_last_60s) / 60.0, 2)

    return HighfreqStatus(
        symbol=symbol,
        snapshot=snapshot,
        minutes_accumulated=int(minutes_accumulated),
        minutes_required=MIN_MINUTES_FOR_TRAINING,
        minutes_remaining=remaining,
        progress_pct=round(progress, 2),
        is_live=is_live,
        has_enough_data=has_enough,
        rows_per_second_estimate=rate,
        server_time=now,
    )


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


# ── DB layer (synchronous, defensive) ─────────────────────────────────────


def _fetch_last_row(db: Session, symbol: str) -> Optional[dict[str, Any]]:
    """Return the most recent ``highfreq_ofi_1s`` row or ``None``.

    Returns ``None`` when the table is empty or does not exist (Phase A
    not yet deployed — the page renders gracefully in that case).
    """
    try:
        row = db.execute(
            text(
                "SELECT ts, symbol, microprice, ofi, depth_imb, spread_bps "
                "FROM highfreq_ofi_1s "
                "WHERE symbol = :symbol "
                "ORDER BY ts DESC LIMIT 1"
            ),
            {"symbol": symbol},
        ).mappings().first()
    except (ProgrammingError, OperationalError) as exc:
        # Table missing / DB unreachable — treat as "not initialised yet".
        logger.warning("highfreq_ofi_1s read failed (%s): %s", symbol, exc)
        return None
    return dict(row) if row is not None else None


def _fetch_minutes_accumulated(db: Session, symbol: str) -> int:
    """How many distinct minute-buckets exist for ``symbol``.

    Used to drive the "data ready in N minutes" countdown. Counted against
    ``highfreq_ofi_1s`` (1 row / sec) rather than ``highfreq_features_1m``
    so the bar moves smoothly even before any minute-aggregation runs.
    """
    try:
        result = db.execute(
            text(
                "SELECT COUNT(DISTINCT date_trunc('minute', ts)) AS n "
                "FROM highfreq_ofi_1s WHERE symbol = :symbol"
            ),
            {"symbol": symbol},
        ).scalar()
    except (ProgrammingError, OperationalError) as exc:
        logger.warning("highfreq_ofi_1s minute-count failed (%s): %s", symbol, exc)
        return 0
    return int(result or 0)


def _fetch_rows_last_60s(db: Session, symbol: str) -> Optional[int]:
    """Count of 1-second rows in the last 60 s — proxy for ingest health."""
    try:
        result = db.execute(
            text(
                "SELECT COUNT(*) AS n FROM highfreq_ofi_1s "
                "WHERE symbol = :symbol AND ts > now() - interval '60 seconds'"
            ),
            {"symbol": symbol},
        ).scalar()
    except (ProgrammingError, OperationalError):
        return None
    return int(result or 0)


# Forecast endpoint reads enough seconds of history to assemble at
# least one COMPLETE 1-minute feature bar (60s) plus margin so the
# build_latest_feature_row() helper can drop the in-flight current minute.
# 180s = ~3 minutes = comfortable headroom even right after a reconnect.
_FORECAST_LOOKBACK_SECONDS: int = 180


def _fetch_recent_seconds(
    db: Session, symbol: str, lookback_seconds: int = _FORECAST_LOOKBACK_SECONDS,
) -> Optional[pd.DataFrame]:
    """Fetch the last ``lookback_seconds`` of 1-s rows for ``symbol``.

    Returns
    -------
    pd.DataFrame
        Columns matching ``highfreq_ofi_1s`` (ts, symbol, ofi, microprice,
        depth_imb, spread_bps, trade_imb, n_updates). Sorted by ``ts``
        ascending. May be empty if the ingestor is down.
    None
        If the query failed (DB unreachable / table missing). Caller
        should surface this as a 503, distinct from "model loaded but
        nothing to score on".
    """
    try:
        rows = db.execute(
            text(
                "SELECT ts, symbol, ofi, microprice, depth_imb, "
                "spread_bps, trade_imb, n_updates "
                "FROM highfreq_ofi_1s "
                "WHERE symbol = :symbol "
                "  AND ts > now() - make_interval(secs => :secs) "
                "ORDER BY ts ASC"
            ),
            {"symbol": symbol, "secs": int(lookback_seconds)},
        ).mappings().all()
    except (ProgrammingError, OperationalError) as exc:
        logger.warning("highfreq_ofi_1s recent-rows fetch failed (%s): %s", symbol, exc)
        return None
    return pd.DataFrame([dict(r) for r in rows])


# ── Router ────────────────────────────────────────────────────────────────

# Templates dir is repo-root/templates; resolved relative to this file so the
# router works whether the FastAPI app is launched from the repo root or
# anywhere else (tests, IDEs).
_TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "templates",
)
templates = Jinja2Templates(directory=_TEMPLATES_DIR)


router = APIRouter(tags=["highfreq"])


def _get_db():
    """SQLAlchemy session dependency — mirrors ``app.main.get_db``.

    Deferred import: ``app.db`` requires ``DATABASE_URL`` at import time,
    so we delay until the first request. This keeps the module importable
    under pytest without a live database.
    """
    from app.db import SessionLocal  # noqa: WPS433 — intentional lazy import

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/highfreq", response_class=HTMLResponse)
async def highfreq_page(request: Request) -> HTMLResponse:
    """Render the live ``/highfreq`` page.

    The HTML is intentionally minimal — actual data is filled in client-side
    by polling :func:`get_status` every 2 seconds. This keeps server-side
    rendering cheap and lets the page degrade gracefully if the ingest
    service is briefly down.
    """
    # Note: starlette ≥ 0.27 deprecated the old `TemplateResponse(name, {"request": request, ...})`
    # signature; in starlette ≥ 0.40 (which the Tokyo venv ships) it raises
    # `TypeError: unhashable type: 'dict'` outright. The new API takes
    # ``request`` as the first positional arg with a separate ``context`` dict.
    return templates.TemplateResponse(
        request,
        "highfreq.html",
        {
            "symbol": DEFAULT_SYMBOL,
            "minutes_required": MIN_MINUTES_FOR_TRAINING,
        },
    )


@router.get("/api/highfreq/status")
async def get_status(
    symbol: str = DEFAULT_SYMBOL,
    db: Session = Depends(_get_db),
) -> JSONResponse:
    """Return the current ingest snapshot + training-readiness countdown."""
    symbol = symbol.upper()
    last_row = _fetch_last_row(db, symbol)
    minutes_accumulated = _fetch_minutes_accumulated(db, symbol)
    rows_last_60s = _fetch_rows_last_60s(db, symbol)

    payload = assemble_status(
        symbol=symbol,
        last_row=last_row,
        minutes_accumulated=minutes_accumulated,
        rows_last_60s=rows_last_60s,
        now=datetime.now(tz=timezone.utc),
    )
    return JSONResponse(content=payload.to_dict())


@router.get("/api/highfreq/health")
async def get_health(
    symbol: str = DEFAULT_SYMBOL,
    db: Session = Depends(_get_db),
) -> JSONResponse:
    """Liveness probe: 200 if rows are flowing, 503 otherwise.

    Used by uptime monitors and the systemd watchdog. Cheap (one indexed
    query) so it can be polled at high frequency without DB pressure.
    """
    symbol = symbol.upper()
    rows_last_60s = _fetch_rows_last_60s(db, symbol)

    if rows_last_60s is None:
        # DB or table missing — still report what we know without crashing.
        return JSONResponse(
            status_code=503,
            content={"ok": False, "reason": "database_unavailable", "symbol": symbol},
        )

    if rows_last_60s == 0:
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "reason": "no_rows_in_last_60s",
                "symbol": symbol,
                "rows_last_60s": 0,
            },
        )

    return JSONResponse(
        content={
            "ok": True,
            "symbol": symbol,
            "rows_last_60s": rows_last_60s,
        }
    )


# ── Forecast endpoint (Phase B scaffold) ──────────────────────────────────


def _get_forecast_predictor() -> LivePredictor:
    """DI-friendly predictor accessor — overrideable in tests."""
    return get_predictor()


@router.get("/api/highfreq/forecast")
async def get_forecast(
    symbol: str = DEFAULT_SYMBOL,
    db: Session = Depends(_get_db),
    predictor: LivePredictor = Depends(_get_forecast_predictor),
) -> JSONResponse:
    """Latest 1-minute directional forecast for ``symbol``.

    Returns 200 with ``prob_up`` once the trainer has produced a
    ``.cbm`` AND there's a complete recent 1-minute bar to score on.
    Until then, returns 503 with a structured ``reason``:

    * ``no_model_yet`` — trainer hasn't run yet / no weights file
    * ``database_unavailable`` — DB unreachable
    * ``not_enough_recent_data`` — no complete 1-minute bar in the
      recent window (cold-start, or reconnect just happened)

    The response always includes the predictor ``model`` block so
    clients can show "model age" / "calibrated?" badges even on 503.
    """
    symbol = symbol.upper()
    status = predictor.status()

    # Branch 1: no model on disk → trainer hasn't shipped weights yet.
    if not status.has_model:
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "reason": "no_model_yet",
                "symbol": symbol,
                "model": status.to_dict(),
                "ts": datetime.now(tz=timezone.utc).isoformat(),
            },
        )

    # Branch 2: model loaded but DB unreachable (e.g. ingest restart).
    df_seconds = _fetch_recent_seconds(db, symbol)
    if df_seconds is None:
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "reason": "database_unavailable",
                "symbol": symbol,
                "model": status.to_dict(),
                "ts": datetime.now(tz=timezone.utc).isoformat(),
            },
        )

    # Branch 3: not enough recent data for a complete bar (cold-start /
    # reconnect window). build_latest_feature_row drops the in-flight
    # current minute, so we need at least one previous COMPLETE minute.
    feature_row = build_latest_feature_row(df_seconds)
    if feature_row is None:
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "reason": "not_enough_recent_data",
                "symbol": symbol,
                "model": status.to_dict(),
                "rows_seen": int(len(df_seconds)),
                "ts": datetime.now(tz=timezone.utc).isoformat(),
            },
        )

    prob_up = predictor.predict(feature_row)
    if prob_up is None:
        # Defensive: status said has_model=True but predict returned None.
        # Treat as transient — return 503 so the caller retries.
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "reason": "model_unavailable",
                "symbol": symbol,
                "model": status.to_dict(),
                "ts": datetime.now(tz=timezone.utc).isoformat(),
            },
        )

    # Map probability to a human-readable signal so the UI doesn't have to
    # re-derive thresholds. Same convention as the future paper-trader:
    # prob > 0.55 = "up" tilt, prob < 0.45 = "down" tilt, else "neutral".
    if prob_up >= 0.55:
        signal = "up"
    elif prob_up <= 0.45:
        signal = "down"
    else:
        signal = "neutral"

    return JSONResponse(
        content=_scrub({
            "ok": True,
            "symbol": symbol,
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "horizon_minutes": 1,
            "prob_up": float(prob_up),
            "signal": signal,
            "calibrated": bool(status.is_calibrated),
            "model": status.to_dict(),
        })
    )


# ── Paper trades endpoint (Phase C UI block) ──────────────────────────────


# Default page size for the recent-trades list. Big enough that the UI's
# 24h sparkline + recent table don't both need separate queries; small
# enough that even after months of trading the response is a few KB.
DEFAULT_PAPER_TRADES_LIMIT: int = 50

# Hard cap to prevent a `?limit=99999999` browser-bug from melting the DB.
MAX_PAPER_TRADES_LIMIT: int = 500


def _fetch_recent_paper_trades(
    db: Session, symbol: str, limit: int,
) -> Optional[list[dict[str, Any]]]:
    """Return the ``limit`` most recent closed paper trades for ``symbol``.

    ``None`` distinguishes "DB unreachable / table missing" (caller
    surfaces 503) from "no trades yet" (caller returns empty list — a
    valid 200).
    """
    try:
        rows = db.execute(
            text(
                "SELECT id, symbol, side, qty, "
                "entry_ts, entry_price, entry_prob_up, "
                "exit_ts, exit_price, exit_reason, "
                "fee_paid_total_usd, pnl_usd, pnl_bps, "
                "model_version, written_at "
                "FROM paper_trades "
                "WHERE symbol = :symbol "
                "ORDER BY exit_ts DESC "
                "LIMIT :limit"
            ),
            {"symbol": symbol, "limit": int(limit)},
        ).mappings().all()
    except (ProgrammingError, OperationalError) as exc:
        logger.warning("paper_trades fetch failed (%s): %s", symbol, exc)
        return None
    return [dict(r) for r in rows]


def _fetch_paper_trades_total(db: Session, symbol: str) -> Optional[int]:
    """Lifetime trade count (for the "X trades total" badge). ``None``
    on DB error — caller treats as unknown without 503-ing the page."""
    try:
        return int(db.execute(
            text("SELECT COUNT(*) FROM paper_trades WHERE symbol = :symbol"),
            {"symbol": symbol},
        ).scalar() or 0)
    except (ProgrammingError, OperationalError):
        return None


def compute_paper_stats(
    *,
    trades_recent: list[dict[str, Any]],
    total_trades_lifetime: Optional[int],
    now: datetime,
    config: PaperTraderConfig,
    risk_caps: RiskCaps,
) -> dict[str, Any]:
    """Pure function — derive the stats block shown above the trades table.

    Designed to mirror what ``PaperTrader.status()`` would compute in
    the runner process, but reconstructed from history. There's a
    small race window when the runner has just opened a position (the
    DB doesn't see it until close) — the UI accepts that and labels
    the *open* position state via a separate snapshot endpoint if we
    ever need it. For Phase C this is fine: the UI is a *read-only
    log*, not a control surface.

    ``trades_recent`` must be sorted DESC by ``exit_ts`` (matches what
    :func:`_fetch_recent_paper_trades` returns).
    """
    # "Today" is computed in UTC since exit_ts is TIMESTAMPTZ stored as UTC.
    today_utc = now.astimezone(timezone.utc).date()

    trades_today = [
        t for t in trades_recent
        if t["exit_ts"].astimezone(timezone.utc).date() == today_utc
    ]
    today_pnl_usd = sum(float(t["pnl_usd"]) for t in trades_today)
    wins_today = sum(1 for t in trades_today if float(t["pnl_usd"]) >= 0)
    losses_today = len(trades_today) - wins_today

    # Consecutive losses ending at the most recent trade. Walk newest →
    # oldest; stop at the first non-losing trade.
    consecutive_losses = 0
    for t in trades_recent:
        if float(t["pnl_usd"]) < 0:
            consecutive_losses += 1
        else:
            break

    halted_reason: Optional[str] = None
    if consecutive_losses >= risk_caps.max_consecutive_losses:
        halted_reason = "loss_streak"
    elif today_pnl_usd <= -risk_caps.max_daily_loss_usd:
        halted_reason = "daily_loss"

    last_trade_ts = trades_recent[0]["exit_ts"] if trades_recent else None
    latest_model_version = (
        trades_recent[0]["model_version"] if trades_recent else None
    )

    # Cumulative P&L for the last N trades (newest first → reverse to
    # plot left-to-right). One pass; cheap.
    cumulative_pnl_series: list[dict[str, Any]] = []
    running = 0.0
    for t in reversed(trades_recent):
        running += float(t["pnl_usd"])
        cumulative_pnl_series.append({
            "exit_ts": t["exit_ts"].isoformat(),
            "cum_pnl_usd": round(running, 6),
        })

    return {
        "total_trades": total_trades_lifetime,
        "trades_today": len(trades_today),
        "today_pnl_usd": round(today_pnl_usd, 4),
        "wins_today": wins_today,
        "losses_today": losses_today,
        "consecutive_losses": consecutive_losses,
        "halted_reason": halted_reason,
        "last_trade_ts": last_trade_ts.isoformat() if last_trade_ts else None,
        "latest_model_version": latest_model_version,
        "cumulative_pnl_series": cumulative_pnl_series,
    }


def _serialise_trade(t: dict[str, Any]) -> dict[str, Any]:
    """asyncpg/sqlalchemy give us datetimes — JSON-encode them as ISO."""
    out = dict(t)
    for k in ("entry_ts", "exit_ts", "written_at"):
        v = out.get(k)
        if isinstance(v, datetime):
            out[k] = v.isoformat()
    # Decimal → float for JSON.
    for k in (
        "qty", "entry_price", "entry_prob_up",
        "exit_price", "fee_paid_total_usd", "pnl_usd", "pnl_bps",
    ):
        v = out.get(k)
        if v is not None:
            out[k] = float(v)
    return out


@router.get("/api/highfreq/paper_trades")
async def get_paper_trades(
    symbol: str = DEFAULT_SYMBOL,
    limit: int = DEFAULT_PAPER_TRADES_LIMIT,
    db: Session = Depends(_get_db),
) -> JSONResponse:
    """Recent closed paper trades + computed running stats.

    Returns 200 always (never 503): empty trades list is a valid state
    during the cold-start period before the trainer ships a model.
    Database unavailability degrades to ``trades=[]`` with a structured
    ``db_status`` flag — the UI can show "DB hiccup" without losing the
    page entirely.
    """
    symbol = symbol.upper()
    limit = max(1, min(int(limit), MAX_PAPER_TRADES_LIMIT))

    config = PaperTraderConfig()
    risk_caps = RiskCaps()

    trades_raw = _fetch_recent_paper_trades(db, symbol, limit)
    if trades_raw is None:
        # DB unreachable — return a soft-failed 200 so the UI keeps
        # rendering the rest of the page.
        return JSONResponse(content={
            "ok": False,
            "db_status": "unavailable",
            "symbol": symbol,
            "trades": [],
            "stats": None,
            "config": _config_to_dict(config, risk_caps),
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        })

    total_lifetime = _fetch_paper_trades_total(db, symbol)
    now = datetime.now(tz=timezone.utc)

    stats = compute_paper_stats(
        trades_recent=trades_raw,
        total_trades_lifetime=total_lifetime,
        now=now,
        config=config,
        risk_caps=risk_caps,
    )

    return JSONResponse(content=_scrub({
        "ok": True,
        "db_status": "ok",
        "symbol": symbol,
        "trades": [_serialise_trade(t) for t in trades_raw],
        "stats": stats,
        "config": _config_to_dict(config, risk_caps),
        "ts": now.isoformat(),
    }))


def _config_to_dict(
    config: PaperTraderConfig, risk_caps: RiskCaps,
) -> dict[str, Any]:
    """Expose the trader's tunables to the UI so it can label thresholds
    correctly (e.g. "halted: ≥5 consecutive losses")."""
    return {
        "entry_long_threshold": config.entry_long_threshold,
        "entry_short_threshold": config.entry_short_threshold,
        "horizon_minutes": config.horizon_minutes,
        "max_qty_usd": config.max_qty_usd,
        "maker_fee_bps_per_side": config.maker_fee_bps_per_side,
        "max_consecutive_losses": risk_caps.max_consecutive_losses,
        "max_daily_loss_usd": risk_caps.max_daily_loss_usd,
    }


# ── Training report endpoint (Phase C.5 calibration progress widget) ──────


# Minimum minutes (post-neutral-band drop) for one walk-forward fold.
# Mirrors the trainer's check; surfaced to the UI so users can see
# "X / Y bars accumulated" without inspecting metrics.json by hand.
MIN_BARS_FOR_FIRST_FOLD: int = 1500

# Path to the metrics JSON the trainer writes alongside the .cbm.
# Overridable via env so the same web app can serve dev / prod.
DEFAULT_METRICS_PATH = Path(
    os.getenv("HIGHFREQ_METRICS_PATH", "weights/highfreq/btcusdt_1m_metrics.json")
)


@router.get("/api/highfreq/training_report")
async def get_training_report() -> JSONResponse:
    """Return the trainer's last metrics report + computed fold-readiness.

    Used by the UI to render two things: (a) the calibration status
    badge ("dir_acc 0.547 [0.521, 0.572]" or "low directional skill"),
    (b) the "X / Y bars for next fold" progress widget — so the user
    sees ramp-up progress visually rather than digging into journals.

    Returns 200 with ``ok=False, reason="no_report_yet"`` if the
    trainer hasn't written its first ``metrics.json`` yet (cold-start).
    Never 503 — the page must keep rendering through trainer outages.
    """
    path = DEFAULT_METRICS_PATH
    if not path.exists():
        return JSONResponse(content={
            "ok": False,
            "reason": "no_report_yet",
            "metrics_path": str(path),
            "min_bars_for_first_fold": MIN_BARS_FOR_FIRST_FOLD,
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        })

    try:
        report = json.loads(path.read_text())
    except Exception as exc:
        logger.warning("training_report read failed: %s", exc)
        return JSONResponse(content={
            "ok": False,
            "reason": "report_unreadable",
            "metrics_path": str(path),
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        })

    # Computed fold readiness (UI can show "726 / 1500 bars (48%)").
    bars_after_drop = int(report.get("n_minutes_after_neutral_drop") or 0)
    fold_ready_pct = min(100.0, 100.0 * bars_after_drop / MIN_BARS_FOR_FIRST_FOLD)

    # Age of the report — UI shows "trained 4 hours ago" so users can
    # tell stale weights from fresh ones.
    try:
        report_age_seconds = max(
            0.0, datetime.now(tz=timezone.utc).timestamp() - path.stat().st_mtime,
        )
    except OSError:
        report_age_seconds = None

    return JSONResponse(content=_scrub({
        "ok": True,
        "report": report,
        "report_age_seconds": report_age_seconds,
        "min_bars_for_first_fold": MIN_BARS_FOR_FIRST_FOLD,
        "fold_ready_pct": round(fold_ready_pct, 1),
        "ts": datetime.now(tz=timezone.utc).isoformat(),
    }))
