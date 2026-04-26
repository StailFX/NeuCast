"""Async loop that drives the paper trader once per minute on bar close.

This is the **runtime glue** between the four already-tested layers:

    feature_pipeline.build_latest_feature_row   ← reads highfreq_ofi_1s
                |
                v
    predictor.LivePredictor.predict             ← reads weights/highfreq/*.cbm
                |
                v
    paper_trader.PaperTrader.on_bar_close       ← state machine
                |
                v
    write_paper_trade  → INSERT paper_trades    ← this module's I/O

The runner intentionally has **no business logic** beyond orchestration —
all decisions live in the layers above, which are 100 % unit-tested.
What this module gets right or wrong is timing (when to fire), error
handling (one bad iteration must not kill the loop), and graceful
shutdown (force-close any open position so we don't strand state).

Lifecycle
---------

The loop sleeps until the next minute boundary + a small grace period
(``BAR_GRACE_SECONDS``), then:

1. Fetches the last ~3 minutes of 1-second rows from Postgres.
2. Builds the latest COMPLETE-minute feature row (drops the in-flight
   current minute — see ``feature_pipeline.build_latest_feature_row``).
3. Calls ``predictor.predict`` for ``prob_up``.
4. Calls ``trader.on_bar_close``. If a trade closes on this tick, the
   returned :class:`PaperTrade` is inserted into the ``paper_trades``
   table via :func:`write_paper_trade`.
5. Logs status snapshot every ``STATUS_LOG_EVERY_N_TICKS`` minutes.

Cold-start behaviour
--------------------

The runner is safe to start BEFORE the trainer has produced a model:

* ``predictor.status().has_model == False`` → ``predict`` returns ``None``
  → runner skips the iteration with ``"no model yet — skipping"`` log.
* Even with a model, ``trader.on_bar_close`` will short-circuit on the
  calibration gate (``SKIP_NOT_CALIBRATED``) until the metrics JSON
  shows ``dir_acc_ci_low > 0.50``.

This means we can deploy the runner today on Tokyo and let it sit
quietly until the model arrives. No code change needed when the trainer
finally ships its first ``.cbm`` — the loop will start opening trades
on the next minute boundary.

Graceful shutdown
-----------------

On ``SIGTERM`` (systemd stop) or ``SIGINT`` (Ctrl-C), the loop exits
its sleep, calls ``trader.force_close`` to flatten any open position,
persists the resulting trade, and shuts down the asyncpg pool cleanly.
This avoids stranding open positions in ``trader.state.open_position``
between restarts.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import asyncpg
import pandas as pd

from app.highfreq.feature_pipeline import build_latest_inference_bar
from app.highfreq.paper_trader import (
    PaperTrade,
    PaperTrader,
    PaperTraderConfig,
    RiskCaps,
)
from app.highfreq.predictor import LivePredictor

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Tunables
# ────────────────────────────────────────────────────────────────────────────


# Grace period after the minute boundary before we read the DB. The
# aggregator emits at 1-second cadence and `flush_batch=1` means each
# row is written immediately — but there's still ~50-100 ms between the
# WS frame arriving at Tokyo and Postgres acknowledging the INSERT.
# 1.5 s comfortably covers that without making the runner feel laggy.
BAR_GRACE_SECONDS: float = 1.5

# How far back to read 1-s rows when building the latest feature row.
# We need ≥ 60 s of the last complete minute, plus headroom for the
# in-flight minute that gets dropped. 180 s = three minutes is generous.
LOOKBACK_SECONDS: int = 180

# How often (in ticks ≡ minutes) to log a full status snapshot, on top
# of the per-iteration "decision" log line. Every 5 minutes keeps the
# journal readable but still gives operational visibility.
STATUS_LOG_EVERY_N_TICKS: int = 5

# asyncpg pool sizing — 2 connections is plenty (one for fetch, one for
# write), with min_size=1 so we don't hold a connection idle when there's
# nothing to do.
POOL_MIN_SIZE: int = 1
POOL_MAX_SIZE: int = 2

# Default symbol — overridable via HIGHFREQ_PAPER_SYMBOL env var.
DEFAULT_SYMBOL: str = os.getenv("HIGHFREQ_PAPER_SYMBOL", "BTCUSDT")

# Predictor weights/metrics paths — match the trainer's output paths.
DEFAULT_WEIGHTS_PATH = Path(
    os.getenv("HIGHFREQ_WEIGHTS_PATH", "weights/highfreq/btcusdt_1m.cbm")
)
DEFAULT_METRICS_PATH = Path(
    os.getenv("HIGHFREQ_METRICS_PATH", "weights/highfreq/btcusdt_1m_metrics.json")
)


# ────────────────────────────────────────────────────────────────────────────
# Pure-logic timing helpers (testable without asyncio)
# ────────────────────────────────────────────────────────────────────────────


def next_minute_boundary(now: datetime) -> datetime:
    """Smallest UTC datetime > ``now`` whose seconds and microseconds are 0.

    Used to compute the asyncio.sleep duration between iterations. If
    ``now`` is exactly on the minute (00.000 µs), we still advance to
    the NEXT minute — never zero-sleep, which would burn CPU.
    """
    floored = now.replace(second=0, microsecond=0)
    return floored + timedelta(minutes=1)


def sleep_until_next_tick(now: datetime, grace_seconds: float = BAR_GRACE_SECONDS) -> float:
    """Seconds to sleep until next minute + grace.

    Returns ``float`` so the caller can pass directly to
    ``asyncio.sleep``. Always > 0 (the +grace ensures it).
    """
    target = next_minute_boundary(now) + timedelta(seconds=grace_seconds)
    return max(0.001, (target - now).total_seconds())


def bar_close_ts_for(now: datetime) -> datetime:
    """Conceptual close-time of the bar we're about to score.

    The aggregator labels bars by their START time (e.g. the bar
    starting at 12:34:00 covers 12:34:00 → 12:34:59). It "closes" at
    12:35:00, which is what we pass to ``trader.on_bar_close(ts=…)``
    — that drives the time-stop comparison (elapsed_minutes from
    entry).

    For a runner firing at 12:35:00.5, we want bar_close_ts =
    12:35:00, i.e. ``now`` floored to the minute.
    """
    return now.replace(second=0, microsecond=0)


# ────────────────────────────────────────────────────────────────────────────
# DB I/O
# ────────────────────────────────────────────────────────────────────────────


# SELECT used by the runner to build the latest feature row. Mirrors
# the schema in app/highfreq/migrations/001_initial_schema.sql.
_SELECT_RECENT_ROWS_SQL = """
    SELECT ts, symbol, ofi, microprice, depth_imb,
           spread_bps, trade_imb, n_updates
      FROM highfreq_ofi_1s
     WHERE symbol = $1
       AND ts > now() - make_interval(secs => $2)
     ORDER BY ts ASC
"""

# INSERT for closed paper trades.
_INSERT_PAPER_TRADE_SQL = """
    INSERT INTO paper_trades (
        symbol, side, qty,
        entry_ts, entry_price, entry_prob_up,
        exit_ts, exit_price, exit_reason,
        fee_paid_total_usd, pnl_usd, pnl_bps,
        model_version
    ) VALUES (
        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
    )
    RETURNING id
"""


async def fetch_recent_seconds(
    pool: asyncpg.Pool, symbol: str, lookback_seconds: int = LOOKBACK_SECONDS,
) -> pd.DataFrame:
    """Fetch the last ``lookback_seconds`` of 1-s rows for ``symbol``.

    Returns an empty DataFrame on no rows (NOT None — we want to
    distinguish "DB returned nothing" from "DB unreachable", and the
    latter raises). Caller handles cold-start (empty frame) downstream
    via :func:`feature_pipeline.build_latest_feature_row`.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(_SELECT_RECENT_ROWS_SQL, symbol, lookback_seconds)
    if not rows:
        return pd.DataFrame(columns=[
            "ts", "symbol", "ofi", "microprice", "depth_imb",
            "spread_bps", "trade_imb", "n_updates",
        ])
    return pd.DataFrame([dict(r) for r in rows])


async def write_paper_trade(pool: asyncpg.Pool, trade: PaperTrade) -> int:
    """INSERT one closed trade. Returns the assigned id."""
    async with pool.acquire() as conn:
        new_id = await conn.fetchval(
            _INSERT_PAPER_TRADE_SQL,
            trade.symbol, trade.side, trade.qty,
            trade.entry_ts, trade.entry_price, trade.entry_prob_up,
            trade.exit_ts, trade.exit_price, trade.exit_reason,
            trade.fee_paid_total_usd, trade.pnl_usd, trade.pnl_bps,
            trade.model_version,
        )
    return int(new_id)


# ────────────────────────────────────────────────────────────────────────────
# One-tick processor (testable in isolation)
# ────────────────────────────────────────────────────────────────────────────


async def process_one_tick(
    *,
    pool: asyncpg.Pool,
    predictor: LivePredictor,
    trader: PaperTrader,
    symbol: str,
    now: datetime,
) -> Optional[PaperTrade]:
    """Run one minute-tick of the runner. Returns the closed trade if any.

    Branches:
    * ``predictor.status().has_model is False`` → log + return ``None``.
      The runner stays idle until the trainer ships weights.
    * Feature row is None (cold-start, sparse minute) → log + return ``None``.
    * ``predict`` returns None → log + return ``None``.
    * Trader returns a trade → INSERT, return the trade.
    * Trader returns None (no decision / opened position / skipped) → return ``None``.
    """
    bar_close_ts = bar_close_ts_for(now)
    p_status = predictor.status()

    if not p_status.has_model:
        logger.info(
            "tick %s: no model yet — runner idle (predictor.has_model=False)",
            bar_close_ts.isoformat(),
        )
        return None

    # Pull the recent window.
    df = await fetch_recent_seconds(pool, symbol)
    if df.empty:
        logger.info(
            "tick %s: DB returned no rows in the last %ds — ingest down?",
            bar_close_ts.isoformat(), LOOKBACK_SECONDS,
        )
        return None

    inference = build_latest_inference_bar(df)
    if inference is None:
        logger.info(
            "tick %s: not enough recent data for a complete bar — skipping",
            bar_close_ts.isoformat(),
        )
        return None
    feat, microprice_close = inference

    prob_up = predictor.predict(feat)
    if prob_up is None:
        logger.warning(
            "tick %s: predictor.predict returned None despite has_model=True",
            bar_close_ts.isoformat(),
        )
        return None

    model_version = str(p_status.model_age_seconds or 0)

    trade = trader.on_bar_close(
        ts=bar_close_ts,
        microprice=microprice_close,
        prob_up=float(prob_up),
        calibrated=p_status.is_calibrated,
        model_version=model_version,
    )

    logger.info(
        "tick %s: prob_up=%.4f calibrated=%s last_no_op=%s",
        bar_close_ts.isoformat(),
        prob_up,
        p_status.is_calibrated,
        trader.state.last_no_op.value if trader.state.last_no_op else None,
    )

    if trade is not None:
        try:
            new_id = await write_paper_trade(pool, trade)
            logger.info(
                "PaperTrade #%d written: %s side=%s entry=%.2f exit=%.2f pnl_usd=%.4f reason=%s",
                new_id, trade.symbol, trade.side, trade.entry_price,
                trade.exit_price, trade.pnl_usd, trade.exit_reason,
            )
        except Exception:
            # We logged + computed the trade; if INSERT fails we don't
            # want to lose the trader's accounting (it's already updated
            # state). Re-raise so systemd restarts us — better than
            # silently dropping P&L rows.
            logger.exception("write_paper_trade failed for trade=%s", trade)
            raise

    return trade


# ────────────────────────────────────────────────────────────────────────────
# Main loop
# ────────────────────────────────────────────────────────────────────────────


class _ShutdownFlag:
    """Tiny helper that asyncio.Event would also do — kept explicit for tests."""

    def __init__(self) -> None:
        self._set = False

    def set(self) -> None:
        self._set = True

    def is_set(self) -> bool:
        return self._set


def install_signal_handlers(flag: _ShutdownFlag, loop: asyncio.AbstractEventLoop) -> None:
    """Wire SIGTERM / SIGINT to flip the shutdown flag.

    Uses the loop's add_signal_handler to ensure the flag is flipped
    from the asyncio thread (signal-safe). Falls back to a no-op on
    Windows where add_signal_handler isn't available — irrelevant for
    our Linux-only deployment but keeps the import safe in tests.
    """
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, flag.set)
        except (NotImplementedError, AttributeError):
            # Windows / unusual env — runner will only stop on natural exit.
            logger.debug("signal handler for %s not installable; skipping", sig)


async def run_loop(
    *,
    pool: asyncpg.Pool,
    predictor: LivePredictor,
    trader: PaperTrader,
    symbol: str,
    shutdown: _ShutdownFlag,
    sleep_fn=asyncio.sleep,
    now_fn=lambda: datetime.now(tz=timezone.utc),
) -> None:
    """Main minute-cadence loop.

    ``sleep_fn`` and ``now_fn`` are injectable for tests — pass a
    fast-forward fake to drive N iterations without waiting real wall
    clock time.
    """
    tick_count = 0
    logger.info(
        "paper_trader_runner started: symbol=%s lookback=%ds grace=%.1fs",
        symbol, LOOKBACK_SECONDS, BAR_GRACE_SECONDS,
    )

    while not shutdown.is_set():
        # Sleep until next bar boundary (+ grace).
        sleep_for = sleep_until_next_tick(now_fn())
        await sleep_fn(sleep_for)
        if shutdown.is_set():
            break

        try:
            await process_one_tick(
                pool=pool, predictor=predictor, trader=trader,
                symbol=symbol, now=now_fn(),
            )
        except Exception:
            # One bad tick must not kill the loop. systemd will restart
            # us only on hard process death. Log + continue.
            logger.exception("tick failed (will retry next minute)")

        tick_count += 1
        if tick_count % STATUS_LOG_EVERY_N_TICKS == 0:
            logger.info("status snapshot: %s", trader.status())

    # Graceful shutdown: flatten any open position via force_close.
    logger.info("shutdown requested — checking for open position to force-close")
    if trader.state.open_position is not None:
        # Use the most recent microprice we can get cheaply. If the DB
        # is unavailable here, fall back to the entry price (zero gross
        # P&L recorded by force_close → close logic).
        try:
            df = await fetch_recent_seconds(pool, symbol, lookback_seconds=10)
            last_price = (
                float(df["microprice"].iloc[-1]) if not df.empty
                else trader.state.open_position.entry_price
            )
        except Exception:
            logger.exception("could not fetch last price for force_close; using entry")
            last_price = trader.state.open_position.entry_price

        forced_trade = trader.force_close(
            ts=now_fn(), microprice=last_price, reason="halt_close",
        )
        if forced_trade is not None:
            try:
                new_id = await write_paper_trade(pool, forced_trade)
                logger.info("force-closed trade #%d on shutdown: %s", new_id, forced_trade)
            except Exception:
                logger.exception("write_paper_trade for force_close failed")

    logger.info("paper_trader_runner exited cleanly")


# ────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ────────────────────────────────────────────────────────────────────────────


async def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    database_url = os.environ["DATABASE_URL"]
    symbol = DEFAULT_SYMBOL.upper()

    predictor = LivePredictor(
        weights_path=DEFAULT_WEIGHTS_PATH, metrics_path=DEFAULT_METRICS_PATH,
    )
    trader = PaperTrader(
        symbol,
        config=PaperTraderConfig(),
        risk_caps=RiskCaps(),
    )

    pool = await asyncpg.create_pool(
        dsn=database_url, min_size=POOL_MIN_SIZE, max_size=POOL_MAX_SIZE,
        command_timeout=10.0,
    )

    shutdown = _ShutdownFlag()
    install_signal_handlers(shutdown, asyncio.get_running_loop())

    try:
        await run_loop(
            pool=pool, predictor=predictor, trader=trader,
            symbol=symbol, shutdown=shutdown,
        )
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
