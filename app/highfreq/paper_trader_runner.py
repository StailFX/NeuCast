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
from app.highfreq.predictor import LivePredictor, weights_path_for_symbol
from app.highfreq.signal_telegram import SignalFlipDetector
from app.highfreq import metrics as M


# Module-level singleton — each runner process is one symbol, so this
# detector holds state for exactly one entry. Survives across
# process_one_tick calls; reset to empty on process restart (which is
# the right semantics: a runner restart shouldn't notify on the
# "first" signal it sees, only on flips it observes within its lifetime).
_SIGNAL_FLIP_DETECTOR = SignalFlipDetector()

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
# Multi-symbol deployment uses templated systemd units passing symbol
# via this env (see deploy/neucast-paper-trader@.service).
DEFAULT_SYMBOL: str = os.getenv("HIGHFREQ_PAPER_SYMBOL", "BTCUSDT")


def _weights_path_for(symbol: str, *, horizon_minutes: int = 1) -> Path:
    """Per-(symbol × horizon) .cbm path. Honours explicit
    HIGHFREQ_WEIGHTS_PATH override (single-symbol legacy); otherwise
    derives from the symbol via the trainer's filename convention
    ({symbol.lower()}_{horizon}m.cbm).

    ``horizon_minutes`` defaults to 1 to preserve the production
    contract; set via the ``HF_HORIZON_MINUTES`` env var for
    long-horizon paper traders (release R, 2026-04-29)."""
    explicit = os.getenv("HIGHFREQ_WEIGHTS_PATH")
    if explicit:
        return Path(explicit)
    return weights_path_for_symbol(symbol, horizon_minutes=horizon_minutes)


def _metrics_path_for(weights_path: Path) -> Path:
    """``btcusdt_1m.cbm`` → ``btcusdt_1m_metrics.json`` — same convention
    used by ``predictor._metrics_path_for``."""
    return weights_path.with_suffix("").with_name(weights_path.stem + "_metrics.json")


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

    # Pull the recent window. For multi-horizon we need MORE history:
    # the long-horizon pipeline needs ≥20 complete bars to bootstrap
    # EMA(20) / RSI(14) / Bollinger(20). For a 15m model that means
    # 20 × 15 = 300 minutes; we ask for 25 bars worth = 25 × bar_minutes
    # to also have the in-flight bar's headroom dropped.
    # ``getattr`` defaults preserve the legacy contract for predictor
    # stubs in tests (and any minimal predictor implementation that
    # predates release R's feature_set / bar_minutes accessors).
    fs_for_lookback = getattr(predictor, "feature_set", lambda: "microstructure")()
    bm_for_lookback = getattr(predictor, "bar_minutes", lambda: 1)()
    if fs_for_lookback == "long_horizon" and bm_for_lookback > 1:
        # 25 bars × bar_minutes × 60s + 60s headroom for in-flight drop.
        lookback = 25 * bm_for_lookback * 60 + 60
    else:
        lookback = LOOKBACK_SECONDS
    df = await fetch_recent_seconds(pool, symbol, lookback_seconds=lookback)
    if df.empty:
        logger.info(
            "tick %s: DB returned no rows in the last %ds — ingest down?",
            bar_close_ts.isoformat(), lookback,
        )
        return None

    # Multi-horizon dispatch (release R, 2026-04-29). The predictor
    # reads ``feature_set`` + ``bar_minutes`` from metrics.json and we
    # call the matching live-inference helper so the feature row's
    # column count matches the model's. Legacy 1m / microstructure
    # path (default) is unchanged.  ``getattr`` defaults keep this
    # backwards-compatible with predictor stubs in tests.
    fs = getattr(predictor, "feature_set", lambda: "microstructure")()
    bm = getattr(predictor, "bar_minutes", lambda: 1)()
    if fs == "long_horizon":
        from app.highfreq.feature_pipeline_long_horizon import (
            build_latest_inference_bar_long_horizon,
        )
        inference = build_latest_inference_bar_long_horizon(
            df, bar_minutes=bm,
        )
    else:
        inference = build_latest_inference_bar(df)
    if inference is None:
        logger.info(
            "tick %s: not enough recent data for a complete bar "
            "(feature_set=%s bar_minutes=%d) — skipping",
            bar_close_ts.isoformat(), fs, bm,
        )
        return None
    feat, microprice_close = inference

    # Time the prediction for the latency histogram.
    import time as _time
    _t0 = _time.perf_counter()
    prob_up = predictor.predict(feat)
    M.prediction_latency_seconds.labels(symbol=symbol).observe(
        _time.perf_counter() - _t0,
    )
    if prob_up is None:
        logger.warning(
            "tick %s: predictor.predict returned None despite has_model=True",
            bar_close_ts.isoformat(),
        )
        return None

    # Update calibration gauges for this symbol so the dashboard sees
    # the latest values without needing to wait for the next prediction
    # cycle if calibration just changed.
    M.predictor_calibrated.labels(symbol=symbol).set(
        1.0 if p_status.is_calibrated else 0.0,
    )
    if p_status.dir_acc_ci_low is not None:
        M.predictor_dir_acc_ci_low.labels(symbol=symbol).set(p_status.dir_acc_ci_low)
    if p_status.model_age_seconds is not None:
        M.predictor_model_age_seconds.labels(symbol=symbol).set(p_status.model_age_seconds)

    # Tag trades opened before the model is calibrated with a sentinel
    # version string so the realized-accuracy logger and any cohort
    # analyses can filter them out cleanly. The actual numeric model
    # age still appears in the report's ``model`` block — this just
    # carries a discriminator into the trade row itself.
    if not p_status.is_calibrated and not trader.config.require_calibrated:
        model_version = "pre-calibration-demo"
    else:
        model_version = str(p_status.model_age_seconds or 0)

    # Realized intra-bar vol from the feature row drives vol-adjusted
    # sizing inside the trader. Falls back to None (= fixed notional)
    # if the feature isn't present.
    realized_vol_bps: Optional[float] = None
    try:
        v = float(feat["microprice_high_low_bps"])
        if v > 0 and v < 1e6:  # sanity bounds; bps can't be NaN/inf either
            realized_vol_bps = v
    except (KeyError, ValueError, TypeError):
        pass

    # Classify the signal before we hand it to the trader so the metric
    # reflects what the predictor actually said (independent of whether
    # the trader's halt state allowed action).
    if prob_up >= 0.55:
        signal = "up"
    elif prob_up <= 0.45:
        signal = "down"
    else:
        signal = "neutral"
    M.predictions_total.labels(symbol=symbol, signal=signal).inc()

    # Persist every prediction to predictions_log (idempotent on
    # (ts, symbol)) — even when the trader skips. Drives the UI's
    # signal tape, the Telegram bot replay, and future "model skill
    # vs trader skill" cohorts. Fail-soft: a failed log doesn't
    # affect the bar loop.
    try:
        from app.highfreq.predictions_log import log_prediction_async
        await log_prediction_async(
            pool,
            ts=bar_close_ts,
            symbol=symbol,
            prob_up=float(prob_up),
            signal=signal,
            microprice=float(microprice_close),
            model_version=model_version,
        )
    except Exception:
        logger.warning("predictions_log write failed", exc_info=True)

    # Inline backfill: the row at (ts - 1 min) had its realized_*
    # columns NULL when written. Now we know the t+1 microprice (it
    # IS this bar's close) — fill it. Single UPDATE per tick, no
    # JOIN needed since we already have the value.
    try:
        from app.highfreq.predictions_backfill import backfill_inline_async
        await backfill_inline_async(
            pool,
            symbol=symbol,
            current_ts=bar_close_ts,
            current_microprice=float(microprice_close),
        )
    except Exception:
        logger.warning("predictions_backfill inline failed", exc_info=True)

    # Telegram signal-flip alert (Level 3). Fires on direction change;
    # cold start (first observation since restart) does not notify to
    # avoid spam on systemd-driven restarts.
    prev = _SIGNAL_FLIP_DETECTOR.check_flip(symbol, signal)
    if prev is not None:
        from app.highfreq.signal_telegram import (
            SignalAlertConfig,
            format_flip_message,
            send_signal_alert_async,
        )
        cfg = SignalAlertConfig.from_env()
        if cfg.enabled:
            body = format_flip_message(
                symbol=symbol, old_signal=prev, new_signal=signal,
                prob_up=float(prob_up), microprice=float(microprice_close),
                ts=bar_close_ts, model_version=model_version,
            )
            try:
                await send_signal_alert_async(cfg, body_html=body)
            except Exception:
                logger.warning(
                    "signal_telegram send failed", exc_info=True,
                )

    # Event-aware halt (release L, 2026-04-29). Pause new entries
    # during scheduled high-vol events (FOMC, CPI, ETH forks, etc.).
    # Existing positions still close on time-stop normally — we
    # don't strand capital, just don't open NEW exposure into a
    # known coin-flip window. Calendar lives at
    # docs/highfreq/event_calendar.json; reload-per-tick is cheap
    # (~1 ms for hundreds of entries) and lets the operator update
    # without restarting the runner.
    try:
        from app.highfreq.event_calendar import (
            DEFAULT_CALENDAR_PATH,
            load_events_from_json,
            should_halt_for_event,
        )
        events = load_events_from_json(DEFAULT_CALENDAR_PATH)
        event_decision = should_halt_for_event(
            events, symbol=symbol, now=bar_close_ts,
        )
        if event_decision.halted:
            logger.info(
                "event-halt active for %s: %s @ %s (mins_until=%.1f)",
                symbol, event_decision.event_name,
                event_decision.event_ts_iso,
                event_decision.minutes_until_event or 0.0,
            )
            trader.state.halted_reason = "event"
    except Exception:
        logger.warning("event-calendar check failed", exc_info=True)

    # Anti-skill protection (release I, 2026-04-28). Each tick we
    # query the rolling gross-winrate of the most recent N closed
    # paper trades. If the bootstrap CI's UPPER bound is below 0.5,
    # the model is statistically anti-skilled — meaning it's
    # consistently directional-WRONG and the inverse trade would
    # have been profitable. Three policies via env:
    #   * 'alert'  (default) — Telegram warn, take no action.
    #   * 'halt'   — set the trader's halt flag, block new entries.
    #   * 'invert' — flip prob_up around 0.5 so 'up' becomes 'down'
    #                and vice versa.
    # Fail-soft: any error => default behaviour (no flip, no halt).
    effective_prob_up = float(prob_up)
    try:
        from app.highfreq.anti_skill_detector import (
            fetch_anti_skill_async,
            parse_response_policy,
        )
        anti_skill_report = await fetch_anti_skill_async(pool, symbol=symbol)
        if anti_skill_report.is_anti_skilled:
            policy = parse_response_policy(os.environ.get("HF_ANTI_SKILL_RESPONSE"))
            logger.warning(
                "anti-skill detected for %s (winrate=%.3f, n=%d, policy=%s): %s",
                symbol,
                anti_skill_report.gross_winrate or 0.0,
                anti_skill_report.n_trades_in_window,
                policy,
                anti_skill_report.note,
            )
            if policy == "halt":
                # Use the trader's existing halt mechanism — same
                # codepath as max_consecutive_losses. Setting the
                # reason makes the no-op logger explain itself.
                trader.state.halted_reason = "anti_skill"
            elif policy == "invert":
                # Flip the probability around 0.5 so the entry-side
                # decision inverts. trader.on_bar_close still applies
                # all other gates (calibration, halt, neutral band).
                effective_prob_up = 1.0 - float(prob_up)
            # 'alert' = pass through; the warning log + future
            # Telegram hook is the only response.
    except Exception:
        logger.warning("anti_skill check failed — proceeding without", exc_info=True)

    trade = trader.on_bar_close(
        ts=bar_close_ts,
        microprice=microprice_close,
        prob_up=effective_prob_up,
        calibrated=p_status.is_calibrated,
        model_version=model_version,
        realized_vol_bps=realized_vol_bps,
    )

    # Sync trader state → gauges (simple snapshot, no historical tracking).
    M.paper_consecutive_losses.labels(symbol=symbol).set(
        trader.state.consecutive_losses,
    )
    halt_reason = trader.state.halted_reason or "none"
    # Two-step: clear all halt-state combos for this symbol, then set the
    # current one. Keeps the gauge unambiguous (only one label combination
    # is 1 at a time). Includes "anti_skill" (release I) and "event"
    # (release L) reasons.
    for r in ("loss_streak", "daily_loss", "anti_skill", "event", "none"):
        M.paper_trader_halted.labels(symbol=symbol, reason=r).set(
            1.0 if r == halt_reason else 0.0,
        )

    logger.info(
        "tick %s: prob_up=%.4f calibrated=%s last_no_op=%s",
        bar_close_ts.isoformat(),
        prob_up,
        p_status.is_calibrated,
        trader.state.last_no_op.value if trader.state.last_no_op else None,
    )

    # If the trader OPENED a position this tick, track it. The only
    # observable signal is open_position becoming non-None — but that
    # state could also come from a previous tick. To avoid double-count,
    # check last_no_op == OK_OPENED.
    from app.highfreq.paper_trader import _NoOpReason
    if trader.state.last_no_op == _NoOpReason.OK_OPENED \
            and trader.state.open_position is not None:
        M.paper_trades_opened_total.labels(
            symbol=symbol, side=trader.state.open_position.side,
        ).inc()

    if trade is not None:
        # Counters BEFORE the DB write — even if the write fails, the
        # trader's accounting already counted this trade. The metric
        # should reflect that.
        M.paper_trades_closed_total.labels(
            symbol=symbol, exit_reason=trade.exit_reason,
        ).inc()
        if trade.pnl_usd >= 0:
            M.paper_pnl_usd_total.labels(symbol=symbol).inc(trade.pnl_usd)
        else:
            M.paper_loss_usd_total.labels(symbol=symbol).inc(abs(trade.pnl_usd))

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

        # Refresh the rolling realized-accuracy gauge for both windows.
        # Tiny query (LIMIT 100, indexed (symbol, exit_ts DESC)) — runs
        # every time a trade closes, well under 1 ms. Failures are
        # NEVER fatal for the trader: this is observability, not state.
        from app.highfreq.realized_accuracy import (
            DEFAULT_WINDOW_SIZES,
            fetch_rolling_accuracy,
        )
        for window in DEFAULT_WINDOW_SIZES:
            try:
                snap = await fetch_rolling_accuracy(
                    pool, symbol=symbol, window=window,
                )
                if snap.accuracy is not None:
                    M.paper_realized_accuracy_rolling.labels(
                        symbol=symbol, window=str(window),
                    ).set(snap.accuracy)
                M.paper_realized_trades_in_window.labels(
                    symbol=symbol, window=str(window),
                ).set(snap.n_eligible)
            except Exception:
                logger.warning(
                    "realized_accuracy update failed (symbol=%s window=%d) — "
                    "metric stays at last value", symbol, window,
                    exc_info=True,
                )

        # Telegram trade-close notification. Skips silently when the
        # signal-bot env vars aren't set (config.enabled=False).
        try:
            from app.highfreq.signal_telegram import (
                SignalAlertConfig,
                format_trade_closed_message,
                send_signal_alert_async,
            )
            cfg = SignalAlertConfig.from_env()
            if cfg.enabled:
                body = format_trade_closed_message(
                    symbol=trade.symbol, side=trade.side,
                    entry_price=trade.entry_price,
                    exit_price=trade.exit_price,
                    qty=trade.qty,
                    pnl_usd=trade.pnl_usd, pnl_bps=trade.pnl_bps,
                    exit_reason=trade.exit_reason,
                    entry_ts=trade.entry_ts, exit_ts=trade.exit_ts,
                    model_version=trade.model_version,
                )
                await send_signal_alert_async(cfg, body_html=body)
        except Exception:
            logger.warning(
                "telegram trade-close notification failed", exc_info=True,
            )

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

    # Start a Prometheus /metrics HTTP server. Each templated systemd
    # instance (btcusdt/ethusdt/bnbusdt) gets a different port via env;
    # default 9091 + symbol-index offset. Bound to 127.0.0.1 since
    # Prometheus runs locally on Tokyo and scrapes localhost.
    metrics_port = int(os.getenv("HIGHFREQ_PAPER_METRICS_PORT", "9091"))
    try:
        from prometheus_client import start_http_server
        start_http_server(metrics_port, addr="127.0.0.1")
        logger.info("prometheus metrics server on 127.0.0.1:%d", metrics_port)
    except Exception:
        logger.exception(
            "failed to start prometheus /metrics on :%d — continuing without",
            metrics_port,
        )

    # Multi-horizon support (release R, 2026-04-29). HF_HORIZON_MINUTES
    # selects which trained .cbm to load (default 1 → 1m model);
    # PaperTraderConfig.horizon_minutes mirrors so the time-stop fires
    # at the same horizon the model was trained on.
    horizon_minutes = int(os.environ.get("HF_HORIZON_MINUTES", "1"))
    if horizon_minutes <= 0:
        raise ValueError(
            f"HF_HORIZON_MINUTES must be positive, got {horizon_minutes}"
        )
    weights_path = _weights_path_for(symbol, horizon_minutes=horizon_minutes)
    metrics_path = _metrics_path_for(weights_path)
    logger.info(
        "predictor weights=%s metrics=%s symbol=%s horizon=%dm",
        weights_path, metrics_path, symbol, horizon_minutes,
    )
    predictor = LivePredictor(
        weights_path=weights_path, metrics_path=metrics_path,
    )
    # Demo mode: when ``HF_PAPER_DEMO_MODE=1``, the trader bypasses the
    # ``is_calibrated`` gate and trades on raw P(UP) thresholds even
    # before the model has produced its first walk-forward fold. Trades
    # opened in this mode are tagged ``model_version="pre-calibration-demo"``
    # so they're EXCLUDED from realized-accuracy stats — visible
    # in the UI for activity/visualization, but not counted as
    # evidence of skill. Default off (preserves the original sim-only
    # contract). Document the toggle in /etc/neucast/env.
    demo_mode = os.environ.get("HF_PAPER_DEMO_MODE", "").strip() in ("1", "true", "yes")
    if demo_mode:
        logger.warning(
            "DEMO MODE ENABLED — trader will open positions without "
            "calibration gate. Trades tagged 'pre-calibration-demo' "
            "and excluded from realized-accuracy reports."
        )

    # Tunable entry thresholds via env. Tighter thresholds = fewer
    # trades but stronger signals — cuts the fee-burden noise that
    # dominates retail-tier 1-min directional ML. Defaults preserve
    # the original 0.55 / 0.45 contract; demo mode operators raise
    # to 0.60 / 0.40 to filter out marginal signals.
    entry_long = float(os.environ.get("HF_ENTRY_LONG_THRESHOLD", "0.55"))
    entry_short = float(os.environ.get("HF_ENTRY_SHORT_THRESHOLD", "0.45"))
    if entry_long != 0.55 or entry_short != 0.45:
        logger.info(
            "entry thresholds overridden via env: long >= %.4f, short <= %.4f",
            entry_long, entry_short,
        )

    trader = PaperTrader(
        symbol,
        config=PaperTraderConfig(
            require_calibrated=not demo_mode,
            entry_long_threshold=entry_long,
            entry_short_threshold=entry_short,
            horizon_minutes=horizon_minutes,
        ),
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
