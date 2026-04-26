"""Prometheus metrics for the HFT slice.

Centralised here so:
  * the same Counter/Gauge/Histogram object is referenced from every
    place that updates it (no double-count from re-import);
  * Grafana dashboards have stable, documented metric names and labels;
  * adding a new metric is one line in this file + one line at the
    call site, no scaffolding.

Naming convention follows Prometheus best practice:
  * unit suffix (`_total`, `_seconds`, `_bytes`)
  * `neucast_hf_*` prefix so dashboards can `{__name__=~"neucast_hf_.*"}`
  * label cardinality kept low (`symbol` is the only high-cardinality
    label and it's in {BTCUSDT, ETHUSDT, BNBUSDT})
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ────────────────────────────────────────────────────────────────────────
# Ingest pipeline (app.highfreq.runner + l2_consumer + aggregator)
# ────────────────────────────────────────────────────────────────────────

ws_frames_total = Counter(
    "neucast_hf_ws_frames_total",
    "Total raw frames received from Binance WebSocket (snap+trade combined).",
)

ws_reconnects_total = Counter(
    "neucast_hf_ws_reconnects_total",
    "Number of WebSocket reconnects since process start. >0/hour signals "
    "a degrading connection — alert on this.",
)

snapshots_dispatched_total = Counter(
    "neucast_hf_snapshots_dispatched_total",
    "L2 snapshots delivered to downstream callbacks (aggregator + writer).",
    ["symbol"],
)

trades_dispatched_total = Counter(
    "neucast_hf_trades_dispatched_total",
    "Trades delivered to downstream callbacks.",
    ["symbol"],
)

ofi_rows_written_total = Counter(
    "neucast_hf_ofi_rows_written_total",
    "Per-second OFI feature rows committed to highfreq_ofi_1s.",
    ["symbol"],
)

l2_snapshots_written_total = Counter(
    "neucast_hf_l2_snapshots_written_total",
    "Top-N L2 snapshots committed to highfreq_l2_snapshots.",
    ["symbol"],
)

# ────────────────────────────────────────────────────────────────────────
# Predictor (app.highfreq.predictor)
# ────────────────────────────────────────────────────────────────────────

predictions_total = Counter(
    "neucast_hf_predictions_total",
    "Predictor invocations. Labelled by symbol + outcome (up/down/neutral).",
    ["symbol", "signal"],
)

prediction_latency_seconds = Histogram(
    "neucast_hf_prediction_latency_seconds",
    "Wall time of LivePredictor.predict() — model load + inference.",
    ["symbol"],
    buckets=(0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.0),
)

predictor_calibrated = Gauge(
    "neucast_hf_predictor_calibrated",
    "1 if the loaded model's dir_acc_ci_low > 0.50, else 0.",
    ["symbol"],
)

predictor_dir_acc_ci_low = Gauge(
    "neucast_hf_predictor_dir_acc_ci_low",
    "Lower bound of the latest bootstrap CI on directional accuracy.",
    ["symbol"],
)

predictor_model_age_seconds = Gauge(
    "neucast_hf_predictor_model_age_seconds",
    "Seconds since the loaded .cbm was last modified. Stale model alert.",
    ["symbol"],
)

# ────────────────────────────────────────────────────────────────────────
# Paper trader (app.highfreq.paper_trader_runner)
# ────────────────────────────────────────────────────────────────────────

paper_trades_opened_total = Counter(
    "neucast_hf_paper_trades_opened_total",
    "Paper positions opened. Labelled by symbol + side.",
    ["symbol", "side"],
)

paper_trades_closed_total = Counter(
    "neucast_hf_paper_trades_closed_total",
    "Paper positions closed. Labelled by symbol + exit_reason.",
    ["symbol", "exit_reason"],
)

paper_pnl_usd_total = Counter(
    "neucast_hf_paper_pnl_usd_total",
    "Cumulative paper P&L in USD (only positive when winning trades). "
    "Negative pnl uses a separate counter — Counter must monotonically "
    "increase for prometheus rate() to work.",
    ["symbol"],
)

paper_loss_usd_total = Counter(
    "neucast_hf_paper_loss_usd_total",
    "Cumulative paper losses in USD (sum of |pnl| for losing trades).",
    ["symbol"],
)

paper_trader_halted = Gauge(
    "neucast_hf_paper_trader_halted",
    "1 if the trader is halted on risk caps, else 0.",
    ["symbol", "reason"],
)

paper_consecutive_losses = Gauge(
    "neucast_hf_paper_consecutive_losses",
    "Current consecutive-loss streak per symbol.",
    ["symbol"],
)

paper_realized_accuracy_rolling = Gauge(
    "neucast_hf_paper_realized_accuracy_rolling",
    "Rolling realized directional accuracy of the paper trader, per "
    "(symbol, window_size). 'window_size' counts trades not minutes — "
    "labels are the rolling N (e.g. '50','100'). Updated whenever a "
    "trade closes via :mod:`app.highfreq.realized_accuracy`. Compares "
    "side ('long' vs 'short') against the realized exit-vs-entry "
    "direction; halt_close exits are excluded from the sample because "
    "they're forced by risk caps and don't reflect model skill.",
    ["symbol", "window"],
)

paper_realized_trades_in_window = Gauge(
    "neucast_hf_paper_realized_trades_in_window",
    "Number of trades currently inside the rolling-accuracy window "
    "(may be less than the requested N until the trader has run "
    "long enough to fill the window). Labelled by (symbol, window).",
    ["symbol", "window"],
)

# ────────────────────────────────────────────────────────────────────────
# Trainer (app.highfreq.trainer — fired by systemd timer)
# ────────────────────────────────────────────────────────────────────────

trainer_runs_total = Counter(
    "neucast_hf_trainer_runs_total",
    "Trainer runs. Labelled by symbol + outcome (success/no_folds/error).",
    ["symbol", "outcome"],
)

trainer_elapsed_seconds = Histogram(
    "neucast_hf_trainer_elapsed_seconds",
    "Wall time of one trainer run.",
    ["symbol"],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800),
)


def reset_for_tests() -> None:
    """Drop metric state between tests. Internal-only — production
    should never call this."""
    for metric in (
        ws_frames_total, ws_reconnects_total,
        snapshots_dispatched_total, trades_dispatched_total,
        ofi_rows_written_total, l2_snapshots_written_total,
        predictions_total, prediction_latency_seconds,
        paper_trades_opened_total, paper_trades_closed_total,
        paper_pnl_usd_total, paper_loss_usd_total,
        paper_realized_accuracy_rolling, paper_realized_trades_in_window,
        trainer_runs_total, trainer_elapsed_seconds,
    ):
        # prometheus_client doesn't expose a clean reset API; use the
        # internal metric collector. Ok for tests, never in prod.
        try:
            metric._metrics.clear()  # type: ignore[attr-defined]
        except AttributeError:
            pass
    for gauge in (
        predictor_calibrated, predictor_dir_acc_ci_low,
        predictor_model_age_seconds, paper_trader_halted,
        paper_consecutive_losses,
    ):
        try:
            gauge._metrics.clear()  # type: ignore[attr-defined]
        except AttributeError:
            pass
