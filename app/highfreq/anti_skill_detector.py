"""Anti-skill detection — auto-protect against systematically wrong models.

Why this exists
===============

Observed on 2026-04-28 with the BTC/ETH/BNB demo runs:

    retail (+7.5bp): 0W / 16L on each symbol — total -$10.21
    vip9   (0.0bp):  7W /  9L on BTC      — gross winrate 43.75%

A 43.75 % gross winrate on 16 trades is **not** statistically distinct
from chance (one-sided p ≈ 0.31), but the *directional pattern* — fee-
unadjusted winrate consistently below 50 % — is the canonical signature
of a model with **inverted edge**. At small sample it's noise; at large
sample it'd mean the predictor is reading short-term momentum signals
that mean-revert at the 1-minute horizon, so going LONG when the model
says ``up`` is systematically wrong.

This module surfaces that condition and gives the operator three
escape hatches:

1. **alert** — emit a Telegram warning, take no action. Default.
   Pure observability: gives the operator visibility into the
   condition without changing trading behaviour.

2. **halt** — set the trader's halt flag the same way ``max_consecutive_losses``
   does. New positions blocked until UTC midnight or manual reset.
   Conservative: stops bleeding without claiming "we know how to
   profit from anti-skill."

3. **invert** — flip the side decision (``long`` ↔ ``short``). If
   the model is genuinely anti-skilled, this captures the inverted
   edge. **Aggressive**: assumes the anti-skill is structural, not
   regime-shift transient. Only enable after the **alert** mode has
   fired stably for days.

The mode is selected via ``HF_ANTI_SKILL_RESPONSE`` env (alert /
halt / invert; default alert).

How "anti-skill" is detected
----------------------------

We compute *gross* winrate (P&L > 0 at zero fees) on the most recent
``window`` closed paper trades. Gross — not net — because fees
introduce a deterministic loss bias that's already captured by the
fee-tier sim; what we want is "is the directional CALL right or
wrong?" independent of cost.

Anti-skill fires when **all** of:

* gross winrate ``< threshold`` (default 0.42)
* ``n_trades >= min_sample`` (default 30)
* the bootstrap CI's UPPER bound also ``< 0.50`` — i.e. anti-skill is
  statistically distinguishable from chance, not just point estimate
  in a noisy run

The CI condition is the safety net: at n=30 with point estimate 0.40,
CI is ±0.18, upper 0.58 — crossing 0.50, so we DON'T fire. The
condition fires only when the data is unambiguous.

Defence-grade utility
---------------------

The honest defence story for live-mode anti-skill protection:

> *"At deployment we observed all-losing demo trades. Diagnostics showed
> gross winrate 43.75 %, point-estimate slightly below chance. Rather
> than handwave 'small sample noise', I built an anti-skill detector
> that monitors rolling winrate with a bootstrap CI, and gives the
> operator three response policies. The default is alert-only — we
> don't take action on noise. The same machinery would flip to
> 'halt' or 'invert' on real anti-skill if it ever materialised at
> a statistically significant scale."*
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)


# ── Defaults ───────────────────────────────────────────────────────────


DEFAULT_WINDOW: int = 50
"""How many most-recent closed trades feed the detector. 50 ≈ 2 h
of demo-mode trading at typical signal density; long enough to
average out single-bar noise, short enough to react to a genuine
regime shift within a couple of hours."""

DEFAULT_MIN_SAMPLE: int = 30
"""No detection below this. 30 trades give SE ≈ 0.09 on a 0.40
estimate — enough to (with the CI rule) reject H0=chance when the
true rate is ≤ 0.30 or so."""

DEFAULT_THRESHOLD: float = 0.42
"""Gross winrate below which we consider anti-skill plausible. Set
to a notch above pure chance (0.50) shifted by 8 pp — captures the
'consistently wrong' regime without firing on chance."""


# ── Data classes ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class AntiSkillReport:
    """JSON-friendly: emitted by ``compute_anti_skill_from_rows`` and
    rendered by the runner / Telegram bot."""

    symbol: str
    window: int
    min_sample: int
    threshold: float
    n_trades_in_window: int
    n_gross_wins: int
    gross_winrate: float | None
    gross_winrate_ci_low: float | None
    gross_winrate_ci_high: float | None
    is_anti_skilled: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Pure logic ─────────────────────────────────────────────────────────


def _wilson_ci(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95 % CI for a binomial proportion. Same formula as
    ``app.highfreq.web.wilson_ci`` — duplicated here to keep the
    module standalone and importable from anywhere (no web deps)."""
    if n == 0:
        return 0.0, 1.0
    import math
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def _is_gross_win(row: dict[str, Any]) -> bool:
    """A trade is a 'gross win' iff the directional CALL was right —
    independent of fees. Mirrors ``recompute_trade_pnl`` at fee=0."""
    side = row.get("side")
    try:
        entry = float(row["entry_price"])
        exit_p = float(row["exit_price"])
    except (KeyError, ValueError, TypeError):
        return False
    if side == "long":
        return exit_p > entry
    if side == "short":
        return exit_p < entry
    return False


def compute_anti_skill_from_rows(
    rows: Sequence[dict[str, Any]],
    *,
    symbol: str,
    window: int = DEFAULT_WINDOW,
    min_sample: int = DEFAULT_MIN_SAMPLE,
    threshold: float = DEFAULT_THRESHOLD,
) -> AntiSkillReport:
    """Pure: detect anti-skill from already-fetched closed-trade rows.

    ``rows`` MUST be ordered most-recent first (the SQL contract).
    Only the most recent ``window`` are considered. Rows where
    ``exit_reason='halt_close'`` are excluded — those exits were
    forced by risk caps, not directional calls.
    """
    eligible = [r for r in rows if r.get("exit_reason") != "halt_close"][:window]
    n = len(eligible)

    if n < min_sample:
        return AntiSkillReport(
            symbol=symbol,
            window=window,
            min_sample=min_sample,
            threshold=threshold,
            n_trades_in_window=n,
            n_gross_wins=sum(1 for r in eligible if _is_gross_win(r)),
            gross_winrate=(sum(1 for r in eligible if _is_gross_win(r)) / n) if n else None,
            gross_winrate_ci_low=None,
            gross_winrate_ci_high=None,
            is_anti_skilled=False,
            note=f"n={n} < min_sample={min_sample} — detector quiet until more data",
        )

    n_wins = sum(1 for r in eligible if _is_gross_win(r))
    winrate = n_wins / n
    ci_low, ci_high = _wilson_ci(successes=n_wins, n=n)

    # The defence-grade firing rule: point estimate AND CI upper bound
    # both below threshold. Belt-and-suspenders against firing on
    # noisy windows where 0.40 winrate is one good streak away from
    # 0.50.
    is_anti_skilled = (winrate < threshold) and (ci_high < 0.50)

    if is_anti_skilled:
        note = (
            f"ANTI-SKILL DETECTED: gross winrate {winrate:.3f} "
            f"< threshold {threshold:.3f}, CI [{ci_low:.3f}, {ci_high:.3f}] "
            f"— upper bound below chance"
        )
    elif winrate < threshold:
        note = (
            f"borderline: point estimate {winrate:.3f} below threshold but "
            f"CI upper {ci_high:.3f} >= 0.50 — within noise of chance"
        )
    elif winrate < 0.50:
        note = (
            f"under chance: winrate {winrate:.3f} but above threshold "
            f"{threshold:.3f} — monitoring"
        )
    else:
        note = f"healthy: gross winrate {winrate:.3f} >= 0.50"

    return AntiSkillReport(
        symbol=symbol,
        window=window,
        min_sample=min_sample,
        threshold=threshold,
        n_trades_in_window=n,
        n_gross_wins=n_wins,
        gross_winrate=winrate,
        gross_winrate_ci_low=ci_low,
        gross_winrate_ci_high=ci_high,
        is_anti_skilled=is_anti_skilled,
        note=note,
    )


# ── Async DB layer (runner) ────────────────────────────────────────────


async def fetch_anti_skill_async(
    pool: Any,
    *,
    symbol: str,
    window: int = DEFAULT_WINDOW,
    min_sample: int = DEFAULT_MIN_SAMPLE,
    threshold: float = DEFAULT_THRESHOLD,
) -> AntiSkillReport:
    """Async fetcher used by the paper-trader runner each tick.

    Tiny query (LIMIT 50, indexed by (symbol, exit_ts DESC)) — well
    under 1 ms. Failures degrade to "no detection" so an asyncpg
    hiccup never fakes anti-skill.
    """
    sql = (
        "SELECT side, entry_price, exit_price, exit_reason, exit_ts "
        "  FROM paper_trades "
        " WHERE symbol = $1 "
        " ORDER BY exit_ts DESC "
        " LIMIT $2"
    )
    try:
        async with pool.acquire() as conn:
            records = await conn.fetch(sql, symbol, int(window))
    except Exception as exc:
        logger.warning("anti_skill fetch failed (symbol=%s): %s", symbol, exc)
        return AntiSkillReport(
            symbol=symbol, window=window, min_sample=min_sample,
            threshold=threshold,
            n_trades_in_window=0, n_gross_wins=0,
            gross_winrate=None,
            gross_winrate_ci_low=None, gross_winrate_ci_high=None,
            is_anti_skilled=False,
            note=f"db_error: {exc!s}",
        )
    rows = [dict(r) for r in records]
    return compute_anti_skill_from_rows(
        rows, symbol=symbol, window=window,
        min_sample=min_sample, threshold=threshold,
    )


# ── Sync DB layer (FastAPI endpoint + Telegram bot) ────────────────────


def fetch_anti_skill_sync(
    db_session: Any,
    *,
    symbol: str,
    window: int = DEFAULT_WINDOW,
    min_sample: int = DEFAULT_MIN_SAMPLE,
    threshold: float = DEFAULT_THRESHOLD,
) -> AntiSkillReport:
    """Sync variant for FastAPI / Telegram bot."""
    from sqlalchemy import text
    sql = text(
        "SELECT side, entry_price, exit_price, exit_reason, exit_ts "
        "  FROM paper_trades "
        " WHERE symbol = :symbol "
        " ORDER BY exit_ts DESC "
        " LIMIT :window"
    )
    res = db_session.execute(sql, {"symbol": symbol, "window": int(window)})
    rows = [dict(r._mapping) for r in res]
    return compute_anti_skill_from_rows(
        rows, symbol=symbol, window=window,
        min_sample=min_sample, threshold=threshold,
    )


# ── Response policy (env-driven) ───────────────────────────────────────


_VALID_RESPONSES: tuple[str, ...] = ("alert", "halt", "invert")


def parse_response_policy(env_value: str | None) -> str:
    """Sanitise ``HF_ANTI_SKILL_RESPONSE`` env value.

    Unrecognised values fall back to ``alert`` (safest) with a warning
    so a typo can't silently disable protection or — worse — flip
    every signal."""
    v = (env_value or "").strip().lower()
    if v in _VALID_RESPONSES:
        return v
    if v:  # set but invalid
        logger.warning(
            "HF_ANTI_SKILL_RESPONSE=%r is not one of %s — falling back to 'alert'",
            v, _VALID_RESPONSES,
        )
    return "alert"
