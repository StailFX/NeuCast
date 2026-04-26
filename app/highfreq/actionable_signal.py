"""Level 4 — actionable trade-signal payload for the UI.

What this is
============

The /highfreq Forecast widget already shows P(UP) + arrow. That's
half the story; the OTHER half is "what would the trader actually do
RIGHT NOW with this signal?" — side, suggested size, entry price,
time horizon, fee estimate, skip reason if any.

This module composes the decision from the latest ``predictions_log``
row + the trader's threshold logic + the position sizer. No DB
mutation; pure read-side. The UI renders this as an actionable
"trade card" next to the Forecast block, so a defence-grade reviewer
can see the *complete* decision pipeline at a glance instead of
piecing together "what does P(up) = 0.42 actually mean for the
trader?".

Pure logic
----------

:func:`compute_decision` takes prediction inputs + config and returns
a JSON-friendly dict — no side effects, fully unit-testable.

The endpoint in :mod:`app.highfreq.web` simply pulls the latest
predictions_log row + builds the dict + returns it. If there's no
prediction yet (cold-start fresh deploy), the endpoint surfaces a
clean "no data yet" rather than 503.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app.highfreq.paper_trader import (
    PaperTraderConfig,
    compute_qty_for_notional,
    decide_entry_side,
)


def compute_decision(
    *,
    prob_up: float,
    microprice: float,
    ts: datetime,
    config: PaperTraderConfig,
    model_version: str,
    calibrated: bool,
    demo_mode: bool,
    halted_reason: str | None = None,
) -> dict[str, Any]:
    """Pure: produce the JSON-friendly decision payload the UI renders.

    ``halted_reason`` is the trader's halt state at the time of the
    decision (None if not halted). When set, ``would_open=False`` and
    ``skip_reason`` reflects the halt — same logic the trader's
    ``on_bar_close`` uses internally, just exposed to the UI.

    The output shape is intentionally flat-ish so the UI can drop it
    straight into a card without nested-property gymnastics.
    """
    # Mirror PaperTrader.on_bar_close gate logic step-by-step.
    skip_reason: str | None = None
    side: str | None = None
    would_open = False

    if halted_reason is not None:
        skip_reason = f"halted_{halted_reason}"
    elif config.require_calibrated and not calibrated:
        skip_reason = "not_calibrated"
    else:
        side = decide_entry_side(prob_up, config)
        if side is None:
            skip_reason = "neutral_signal"
        else:
            would_open = True

    qty_native = (
        compute_qty_for_notional(config.max_qty_usd, microprice)
        if would_open else 0.0
    )

    fee_per_side = (
        qty_native * microprice * (config.maker_fee_bps_per_side / 1e4)
        if would_open else 0.0
    )
    fee_estimate_usd = 2.0 * fee_per_side  # entry + exit, same maker rate

    exit_at = ts + timedelta(minutes=int(config.horizon_minutes))

    return {
        # Inputs the decision was made from.
        "prob_up": float(prob_up),
        "microprice": float(microprice),
        "ts": ts.isoformat(),

        # The decision.
        "would_open": would_open,
        "side": side,
        "skip_reason": skip_reason,

        # Sizing + economics (only meaningful when would_open).
        "suggested_qty_native": float(qty_native),
        "suggested_qty_usd": float(qty_native * microprice) if would_open else 0.0,
        "entry_price": float(microprice) if would_open else None,
        "fee_estimate_usd": float(fee_estimate_usd),
        "time_horizon_min": int(config.horizon_minutes),
        "exit_at_iso": exit_at.isoformat() if would_open else None,

        # Provenance / caveats.
        "model_version": model_version,
        "calibrated": bool(calibrated),
        "demo_mode": bool(demo_mode),
    }


def fetch_latest_prediction_sync(
    db_session: Any, *, symbol: str,
) -> dict[str, Any] | None:
    """Most recent ``predictions_log`` row for ``symbol``, or None
    when nothing has been logged yet (cold-start fresh deploy)."""
    from sqlalchemy import text
    sql = text(
        "SELECT ts, prob_up, signal, microprice, model_version "
        "  FROM predictions_log "
        " WHERE symbol = :symbol "
        " ORDER BY ts DESC "
        " LIMIT 1"
    )
    row = db_session.execute(sql, {"symbol": symbol}).fetchone()
    if row is None:
        return None
    d = dict(row._mapping) if hasattr(row, "_mapping") else dict(row)
    return d
