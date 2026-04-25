# Phase A — End-to-end demo

This file walks through the four moving parts of the HF module — ingest,
training, backtest, UI — with **representative sample output** at each
step. You can run it locally against a fresh Postgres or on the
production VPS once the systemd unit is installed.

If you're reading this as a portfolio artifact: the goal is not for the
sample numbers to be impressive — it's for the *interfaces* (CLI, JSON
shape, page layout) to be legible without running the code yourself.

---

## 0 · Prerequisites

```bash
# Postgres listening on localhost:5433 (matches docker-compose default)
export DATABASE_URL='postgresql://neucast:CHANGE_ME@127.0.0.1:5433/neucast'

# One-time: apply the schema
psql "$DATABASE_URL" < app/highfreq/migrations/001_initial_schema.sql

# Python deps (asyncpg is the only HF-specific add)
pip install -r requirements.txt asyncpg
```

---

## 1 · Start the L2 ingest

```bash
python -m app.highfreq.runner
```

After the first ~10 seconds you should see periodic health logs:

```
2026-04-25 12:00:00,123 [INFO] highfreq.runner: starting highfreq pipeline:
   symbols=['BTCUSDT'] depth=20 speed=100ms flush=5
2026-04-25 12:00:30,456 [INFO] highfreq.runner:
   health: frames=346 snaps=291 trades=55 rows_emitted=32 rows_written=30 reconnects=0
2026-04-25 12:01:00,789 [INFO] highfreq.runner:
   health: frames=689 snaps=583 trades=109 rows_emitted=62 rows_written=60 reconnects=0
```

What each number means:

| Counter | Cadence | Healthy range |
|---|---|---|
| `frames`    | per WS message  | 9-10/sec at @depth20@100ms |
| `snaps`     | per book frame  | ≈ frames |
| `trades`    | per print       | 1-50/sec, depends on volatility |
| `rows_emitted` | per second   | exactly 1/sec |
| `rows_written` | per flush    | lags `rows_emitted` by `flush_batch_size` (default 5) |
| `reconnects` | per WS drop    | should stay at 0 over 24 h+ |

---

## 2 · Open the live UI

```bash
# Local dev
open http://localhost:8100/highfreq

# Production
open https://neucast.ru/highfreq
```

What the page shows:

```
┌──────────────────────────────────────────────────────────────────────┐
│ NeuCast               [● Phase A · Sim only]                         │
│                                                                      │
│ High-Frequency Forecast                                              │
│ Live BTCUSDT ingest from Binance Spot WebSocket — OFI, microprice,  │
│ depth-imbalance на 1-секундной решётке. Тренировка walk-forward     │
│ CatBoost-классификатора стартует автоматически после накопления     │
│ 65 минут данных.                                                     │
│                                                                      │
│ ┌─────────────────────┐  ┌─────────────────────┐  ┌────────────────┐│
│ │ Microprice          │  │ OFI (1s)            │  │ Depth imb.     ││
│ │ 67 524.31    (▲ up) │  │ 1.234               │  │ 0.045          ││
│ └─────────────────────┘  └─────────────────────┘  └────────────────┘│
│ ┌─────────────────────┐  ┌─────────────────────┐  ┌────────────────┐│
│ │ Spread, bps         │  │ Rows / second       │  │ Last update    ││
│ │ 0.62                │  │ 0.97                │  │ 2s ago         ││
│ └─────────────────────┘  └─────────────────────┘  └────────────────┘│
│                                                                      │
│ ┌──────────────────────────────────────────────────────────────────┐│
│ │ Накоплено данных для первого фолда обучения             18.5%  ││
│ │ ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ││
│ │ 12 / 65 минут — осталось 53 мин.                                 ││
│ └──────────────────────────────────────────────────────────────────┘│
│                                                                      │
│ ⚠ Sim-only. Никаких живых ордеров не выставляется (см. ADR-005).   │
│   Все P&L цифры в backtest-репортах — paper trading с явной        │
│   maker/taker fee-моделью (-1bps / +10bps на сторону).              │
└──────────────────────────────────────────────────────────────────────┘
```

Underlying data via JSON:

```bash
curl -s http://localhost:8100/api/highfreq/status | jq .
```

```json
{
  "symbol": "BTCUSDT",
  "snapshot": {
    "ts": "2026-04-25T12:00:58+00:00",
    "symbol": "BTCUSDT",
    "microprice": 67524.31,
    "ofi": 1.234,
    "depth_imb": 0.045,
    "spread_bps": 0.62,
    "age_seconds": 2.0
  },
  "minutes_accumulated": 12,
  "minutes_required": 65,
  "minutes_remaining": 53,
  "progress_pct": 18.46,
  "is_live": true,
  "has_enough_data": false,
  "rows_per_second_estimate": 0.97,
  "server_time": "2026-04-25T12:01:00+00:00"
}
```

---

## 3 · Train the first walk-forward fold

Once `minutes_accumulated ≥ 65` (visible on the UI), run:

```bash
python -m app.highfreq.trainer \
    --symbol BTCUSDT \
    --since-hours 24 \
    --out weights/highfreq/btcusdt_$(date +%Y%m%d).cbm
```

The CLI emits a single JSON report to stdout:

```json
{
  "symbol": "BTCUSDT",
  "n_minutes_total": 1395,
  "n_minutes_after_neutral_drop": 312,
  "n_folds": 6,
  "dir_acc_pooled": 0.5417,
  "dir_acc_ci_low": 0.5031,
  "dir_acc_ci_high": 0.5803,
  "low_directional_skill": false,
  "fold_reports": [
    {"fold": 0, "n_train": 240, "n_test": 60, "dir_acc": 0.5333, "logloss": 0.6884},
    {"fold": 1, "n_train": 300, "n_test": 60, "dir_acc": 0.5500, "logloss": 0.6851},
    "..."
  ],
  "feature_importances": {
    "ofi_sum_1m": 23.4,
    "ofi_mean_1m": 18.7,
    "depth_imb_mean_1m": 14.2,
    "microprice_close_1m": 11.9,
    "spread_bps_mean_1m": 9.6,
    "...": "..."
  },
  "model_path": "weights/highfreq/btcusdt_20260425.cbm",
  "predictions_path": "weights/highfreq/btcusdt_20260425_predictions.parquet"
}
```

Interpreting the result:

* `dir_acc_pooled = 54.2 %` with bootstrap 95 % CI **excluding 50 %**
  → directional signal is detectable. If `dir_acc_ci_low <= 0.5`, the
  trainer flips `low_directional_skill = true` and the operator should
  collect more data before deploying.
* `n_minutes_after_neutral_drop` is the *trainable* sample size — bars
  with `|return| < 1 bp` are dropped per ADR-007.
* `feature_importances` are CatBoost gains, sorted descending. OFI
  features dominate by design (Cont/Kukanov/Stoikov 2014).

---

## 4 · Run the sim-backtest

```bash
python -m app.highfreq.backtest \
    --predictions weights/highfreq/btcusdt_20260425_predictions.parquet \
    --confidence-threshold 0.55 \
    --notional-per-trade 10000 \
    --out reports/highfreq/btcusdt_20260425.json
```

JSON report (truncated):

```json
{
  "config": {
    "confidence_threshold": 0.55,
    "notional_per_trade": 10000.0,
    "fee_model": {"maker_fee_bps": -1.0, "taker_fee_bps": 10.0}
  },
  "taker": {
    "n_trades": 178,
    "win_rate": 0.5337,
    "total_pnl_dollars": -42.18,
    "avg_pnl_bps": -0.24,
    "sharpe_per_minute": -0.018,
    "sharpe_annualised": -13.4,
    "max_drawdown_pct": 0.087
  },
  "maker_at_50pct": {
    "n_trades": 178,
    "win_rate": 0.5337,
    "total_pnl_dollars": 31.46,
    "avg_pnl_bps": 0.18,
    "sharpe_per_minute": 0.013,
    "sharpe_annualised": 9.7,
    "max_drawdown_pct": 0.041
  },
  "maker_at_100pct": {
    "n_trades": 178,
    "win_rate": 0.5337,
    "total_pnl_dollars": 105.10,
    "avg_pnl_bps": 0.59,
    "sharpe_per_minute": 0.044,
    "sharpe_annualised": 32.6,
    "max_drawdown_pct": 0.018
  },
  "fill_rate_sweep": [
    {"maker_fill_rate": 0.0, "total_pnl_dollars": -42.18, "sharpe_annualised": -13.4},
    {"maker_fill_rate": 0.3, "total_pnl_dollars": -8.08,  "sharpe_annualised": -2.9},
    {"maker_fill_rate": 0.5, "total_pnl_dollars": 31.46,  "sharpe_annualised": 9.7},
    {"maker_fill_rate": 0.7, "total_pnl_dollars": 60.91,  "sharpe_annualised": 19.5},
    {"maker_fill_rate": 1.0, "total_pnl_dollars": 105.10, "sharpe_annualised": 32.6}
  ]
}
```

How to read this:

* The same predictions produce **negative P&L as a taker, positive
  P&L as a maker** — the central insight of ADR-005. A backtest that
  reported only one of these would lie about the strategy.
* `fill_rate_sweep` answers "what maker fill rate do you need to break
  even?" — here, ~38 % (linear interpolation between the −2.9 row at
  30 % and the 9.7 row at 50 %).
* `sharpe_annualised` uses the per-minute Sharpe × √525 600. A value
  of 9.7 looks huge by daily-equity-curve standards but is normal for
  a high-Kelly-frequency 1-minute strategy with 178 trades.

---

## 5 · Run the test suite

```bash
python -m pytest tests/ -v
```

```
tests/test_highfreq_backtest.py::test_fee_model_per_side_bps_endpoints PASSED
tests/test_highfreq_backtest.py::test_fee_model_roundtrip_is_2x_per_side PASSED
tests/test_highfreq_backtest.py::test_build_trade_ledger_long_signal_correct_pnl PASSED
...
tests/test_highfreq_trainer.py::test_aggregate_to_minute_basic_shape PASSED
tests/test_highfreq_trainer.py::test_build_target_forward_shift_and_bps PASSED
tests/test_highfreq_trainer.py::test_bootstrap_dir_acc_ci_deterministic PASSED
...
tests/test_highfreq_web.py::test_assemble_status_no_data_yet PASSED
tests/test_highfreq_web.py::test_assemble_status_to_dict_is_json_safe PASSED

============================== 60 passed in 0.68s ==============================
```

The suite is **pure-function** — no Postgres, no FastAPI test client, no
live WebSocket, no model fixtures on disk. This keeps CI cheap and
isolates regressions cleanly to the layer they affect.

---

## 6 · Capturing portfolio screenshots

The UI is intentionally minimal so it screenshots well at 1280×800. The
informative shots are:

1. **Header + live metrics tile** — captures the live-ingest aesthetic.
2. **Countdown bar mid-fill** (e.g. ~50 %) — shows the "data accumulating"
   state and the explicit 65-minute requirement.
3. **JSON status response in browser DevTools** — proves the data layer
   is honest (no inflated numbers).
4. **Backtest JSON output side-by-side with the maker/taker columns** —
   visualises ADR-005 in one image.

Save them under `docs/highfreq/screenshots/` (gitignored — the binaries
are too heavy for the repo; PRs reference them via the Markdown
preview only).
