"""Backfill ``predictions_log.realized_microprice_1m`` and
``realized_correct`` for predictions whose target minute has now
elapsed.

Why
===

When the predictor scores at minute *t*, we don't yet know what the
microprice will be at *t+1*. So ``predictions_log`` is initially
written with ``realized_microprice_1m = NULL``. One minute later the
data IS available — the next minute's bar has closed in
``highfreq_ofi_1s``. This module joins the two and fills in
``realized_correct``:

    correct = (signal == 'up'   AND realized > entry) OR
              (signal == 'down' AND realized < entry)

Neutral signals (``prob_up`` between 0.45 and 0.55) are left as NULL
in ``realized_correct`` — there's no directional claim to verify.
The realized-microprice IS still filled so the UI can render the
post-fact price next to the prediction.

Two run modes
-------------

* :func:`backfill_inline_async` — called from the runner once per
  tick (after writing the new prediction). Updates exactly the row
  for ``ts - 1 minute`` if it's still NULL. Cheap (~ms) and keeps
  the realized columns hot without a separate cron.
* :func:`backfill_window_sync` — sync sweeper for catch-up after a
  runner outage. Runs over all rows in the last *N* hours where
  realized is NULL and the t+1 bar is now available.

Defence-grade utility
---------------------

This is the ONLY way to get model-skill numbers BEFORE the trainer's
walk-forward CV produces folds (which needs 1500+ bars). With
backfill running, after just 30 predictions you can already plot
"realized accuracy by hour" — concrete evidence the system is
producing real-time directionally correct calls (or not).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# SQL — single source of truth
# ──────────────────────────────────────────────────────────────────────

# Fill in realized columns for any prediction in the window whose t+1
# minute now exists in highfreq_ofi_1s. The join is a LATERAL subquery
# averaging microprice over [ts+1m, ts+2m); that mean is what
# aggregate_to_minute would call ``microprice_close`` for the next bar.
#
# We use AVG instead of "last value before ts+2m" because the seconds
# table is dense enough that the mean is within a few cents of the
# last sample, and AVG works in plain SQL without window functions.
_BACKFILL_SQL = """
UPDATE predictions_log p
SET
  realized_microprice_1m = next_bar.realized_close,
  realized_correct = CASE
    WHEN next_bar.realized_close IS NULL                                      THEN NULL
    WHEN p.signal = 'up'   AND next_bar.realized_close > p.microprice         THEN TRUE
    WHEN p.signal = 'down' AND next_bar.realized_close < p.microprice         THEN TRUE
    WHEN p.signal = 'neutral'                                                  THEN NULL
    ELSE FALSE
  END
FROM (
  SELECT
    target.id,
    (
      SELECT AVG(microprice)
        FROM highfreq_ofi_1s
       WHERE symbol = target.symbol
         AND ts >= target.ts + interval '1 minute'
         AND ts <  target.ts + interval '2 minutes'
    ) AS realized_close
    FROM predictions_log target
   WHERE target.realized_microprice_1m IS NULL
     AND target.ts >= now() - (:lookback_hours * interval '1 hour')
     AND target.ts <  now() - interval '1 minute'
     {symbol_filter}
) AS next_bar
WHERE p.id = next_bar.id
  AND next_bar.realized_close IS NOT NULL
"""


# ──────────────────────────────────────────────────────────────────────
# Inline (async) — called by the runner each tick
# ──────────────────────────────────────────────────────────────────────


async def backfill_inline_async(
    pool: Any,  # asyncpg.Pool
    *,
    symbol: str,
    current_ts: datetime,
    current_microprice: float,
) -> int:
    """Fill the realized columns for the row at ``current_ts - 1 min``
    using ``current_microprice`` as the realized close.

    Called from ``paper_trader_runner.process_one_tick`` immediately
    after the new prediction is logged. Avoids a JOIN against
    ``highfreq_ofi_1s`` — we already have the close-minute price in
    hand from the bar we just consumed.

    Returns the number of rows updated (0 or 1).
    """
    target_ts = current_ts - timedelta(minutes=1)
    sql = """
        UPDATE predictions_log
           SET realized_microprice_1m = $3,
               realized_correct = CASE
                 WHEN signal = 'up'   AND $3 > microprice THEN TRUE
                 WHEN signal = 'down' AND $3 < microprice THEN TRUE
                 WHEN signal = 'neutral'                  THEN NULL
                 ELSE FALSE
               END
         WHERE symbol = $1
           AND ts = $2
           AND realized_microprice_1m IS NULL
    """
    try:
        async with pool.acquire() as conn:
            res = await conn.execute(
                sql, symbol, target_ts, float(current_microprice),
            )
        # asyncpg returns "UPDATE N" — parse the count.
        try:
            n = int(res.split()[-1])
        except (ValueError, IndexError):
            n = 0
        return n
    except Exception as exc:
        logger.warning(
            "predictions_backfill inline failed (symbol=%s ts=%s): %s",
            symbol, target_ts, exc,
        )
        return 0


# ──────────────────────────────────────────────────────────────────────
# Window sweeper (sync) — catch-up after runner outage
# ──────────────────────────────────────────────────────────────────────


def backfill_window_sync(
    database_url: str,
    *,
    lookback_hours: int = 6,
    symbol: str | None = None,
) -> int:
    """One-shot sweeper. Fills ALL un-backfilled predictions in the
    last ``lookback_hours`` whose t+1 minute is now in
    ``highfreq_ofi_1s``. Returns rows updated.

    Use cases:
    * Initial backfill after deploying this module — fills all rows
      that were inserted before the inline backfill existed.
    * After a runner outage — the inline backfill missed those minutes;
      the sweeper catches them up.

    Optionally restrict to one ``symbol``; defaults to all.
    """
    from sqlalchemy import create_engine, text

    sql = _BACKFILL_SQL.format(
        symbol_filter="AND target.symbol = :symbol" if symbol else "",
    )
    eng = create_engine(database_url, future=True)
    params: dict[str, Any] = {"lookback_hours": lookback_hours}
    if symbol:
        params["symbol"] = symbol.upper()

    try:
        with eng.begin() as conn:
            result = conn.execute(text(sql), params)
            n = result.rowcount or 0
        logger.info(
            "predictions_backfill window swept: lookback=%dh symbol=%s rows=%d",
            lookback_hours, symbol or "all", n,
        )
        return n
    except Exception as exc:
        logger.warning("predictions_backfill window failed: %s", exc)
        return 0
