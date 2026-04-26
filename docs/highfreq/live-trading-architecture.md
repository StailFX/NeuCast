# Live trading architecture — design proposal

**Status**: design-doc, partially implemented (release C scaffolds the
seams). Real execution path (Binance testnet → mainnet) lands in
follow-up releases gated on calibration evidence.

---

## 1 · Goals and non-goals

**Goals**

* Build the *engineering surface* for live order routing **before** the
  model proves edge, so when the model is ready we can flip a flag
  instead of doing weeks of new work under time pressure.
* Make the paper-trader UI come alive **today** (demo mode, before
  calibration) so reviewers / ourselves see the system end-to-end and
  identify visual / state-machine bugs early.
* Persist the trainer's full output history so we can show
  *time-series* evolution of `dir_acc`, calibration CI, and feature
  importance — defence-grade story is "model improved from 0.51 to
  0.55 over 14 days as data accumulated", not "here's a snapshot".

**Non-goals**

* Mainnet trading anytime soon. Live executor stays a stub with a hard
  env-gate until **all three** of the following land:
  1. Frozen-holdout dir_acc CI lower bound > 0.5 with p < 0.05.
  2. 30+ days of clean Binance Spot Testnet operation (no order drift,
     no reconciliation gaps).
  3. Telegram-bot kill switch + DR drill against the executor.
* Margin / futures. Binance Spot is long-only — short trades on the
  paper-trader assume a synthetic short payoff (we model the P&L as
  if it executed). When (if) we move to live, shorting either requires
  USDM Futures Testnet (separate venue) or stays paper-only on the
  "short" side. This is a deliberate scope cut for the academic phase.

---

## 2 · Three execution modes

The trader interacts with the world through one of three executors,
selected at runtime via `HF_EXECUTOR_KIND`:

```
┌─────────────────────────────────────────────────────────────────┐
│                  app/highfreq/order_executor.py                 │
│                                                                 │
│   ┌──────────────────────────┐   Protocol contract:             │
│   │  OrderExecutor (Protocol)│ ─ open_long  ─ open_short        │
│   └──────────────────────────┘ ─ close_position                 │
│             ▲                     returns ExecutedFill          │
│             │                                                   │
│   ┌─────────┴─────────┬──────────────────────┬────────────────┐ │
│   │                   │                      │                │ │
│   │ SimulatedExecutor │ BinanceTestnetExec   │ BinanceLiveExec│ │
│   │ (default)         │ (stub)               │ (stub+gated)   │ │
│   │                   │                      │                │ │
│   │ fill price = bar  │ POST /api/v3/order   │ same as testnet│ │
│   │ microprice        │ → testnet.binance    │ but mainnet    │ │
│   │ fee = maker_bps   │   .vision            │ + RuntimeError │ │
│   │ no venue id       │ + WS user data       │   if env-gate  │ │
│   │                   │ + reconciliation     │   not set      │ │
│   └───────────────────┴──────────────────────┴────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

The trader doesn't know which one it's holding — it calls
`await executor.open_long(...)` and trusts the returned
`ExecutedFill`.

**Why a Protocol, not subclass-everything**: third-party executors
(Kraken, OKX) plug in structurally without inheriting from our base.
Tests stub it with any duck-typed object.

**What's done in release C** (this release):

* `OrderExecutor` Protocol + `ExecutedFill` dataclass.
* `SimulatedExecutor` matches the trader's existing inline math
  byte-for-byte (cross-checked against `paper_trader.fee_per_side`).
* Stub executors that raise `NotImplementedError` loudly.
* `BinanceLiveExecutor.__post_init__` refuses to construct without
  `HF_LIVE_TRADING_CONFIRMED=1`. Belt-and-suspenders against typo'd
  env vars.

**What's NOT done in release C** (deferred):

* Refactoring `PaperTrader` to *call* the executor. The trader still
  computes fills inline. The executor is a parallel implementation
  for now — proven equivalent by tests, but not yet on the hot path.
  The refactor is a separate PR with full state-machine test coverage
  because changing how positions open/close is the most dangerous
  edit in a trading system.
* Filling in the testnet HTTP / WebSocket client. ~3-5 days of focused
  work; tracked as Release D.

---

## 3 · Demo mode (variant C)

**Problem**: paper-trader sits idle until model is calibrated. The
calibration gate is `dir_acc_ci_low > 0.5` from the trainer's
walk-forward CV — needs ≥ 1500 bars (currently at 1109). Until then,
the paper-trading widgets, P&L sparkline, and recent-trades table
all show empty state. UI looks broken.

**Solution**: opt-in env flag `HF_PAPER_DEMO_MODE=1`. When set:

1. The trader's `require_calibrated` config becomes `False` —
   `on_bar_close` no longer skips entries when `is_calibrated=False`.
2. Trades opened in this mode are tagged `model_version="pre-calibration-demo"`
   so the realized-accuracy logger excludes them from skill stats.
3. A WARN log line on startup so the operator knows the gate is off.
4. UI surfaces a banner ("DEMO MODE — pre-calibration trades shown
   for visualisation only, not counted toward realized accuracy").

**Defence story**: "I added a demo mode so the system pipeline runs
end-to-end before model calibration is proven. Demo trades are
explicitly tagged and excluded from realized-skill statistics. The
real calibration gate remains the trainer's walk-forward CV; demo
mode is for visualisation, not evidence." This is **stronger** than
"system was empty during demo because we waited for calibration" —
shows you understand why the gate matters AND shipped the engineering
to bypass it transparently.

**Wire-up** (release C):

* `paper_trader_runner` reads `HF_PAPER_DEMO_MODE` env, passes
  `require_calibrated=not demo` into `PaperTraderConfig`.
* `paper_trader_runner` constructs `model_version="pre-calibration-demo"`
  for trades opened while uncalibrated.
* `realized_accuracy.fetch_rolling_accuracy` SQL filter:
  `WHERE model_version <> 'pre-calibration-demo'`.

---

## 4 · Trainer cadence + information collection

### Current state

* Trainer fires **daily at 04:00 UTC** per symbol.
* Output: `weights/highfreq/<sym>_1m.cbm` + `weights/highfreq/<sym>_1m_metrics.json`.
* The metrics.json is **overwritten** every run — no history.
* UI / endpoints surface the *current* state only.

### Proposed change: hourly trainer + persistent history

Three changes that work together:

**(a) New cadence: every 4 hours by default, env-overridable**

```
OnCalendar=*-*-* 00,04,08,12,16,20:00:00 UTC
```

Why every 4 hours, not hourly:
* Each run takes ~10 s per symbol on the Tokyo box (3 symbols = ~30 s).
  Hourly = 720 s/day, every 4h = 180 s/day. Difference is negligible
  CPU but the every-4h cadence aligns with the existing L2-archive
  schedule and keeps trainer + archive activity batched.
* If `dir_acc` is genuinely sensitive to ±1 hour of new data, that's
  itself a bad sign (we'd be over-fitting to recent regime). Hourly
  flicker would obscure that signal. Every 4 hours smooths it without
  silently degrading the academic argument.
* Override via `HF_TRAINER_CADENCE_HOURS` for experiments.

**(b) Persistent training-runs history table**

```sql
-- app/highfreq/migrations/004_training_runs.sql
CREATE TABLE IF NOT EXISTS training_runs (
    id                          BIGSERIAL PRIMARY KEY,
    symbol                      TEXT NOT NULL,
    run_started_at              TIMESTAMPTZ NOT NULL,
    elapsed_seconds             DOUBLE PRECISION NOT NULL,

    -- Data inventory at this run
    n_seconds_loaded            INT NOT NULL,
    n_minutes_after_aggregation INT NOT NULL,
    n_minutes_after_neutral_drop INT NOT NULL,

    -- Model evaluation
    n_folds                     INT NOT NULL,
    dir_acc_mean                DOUBLE PRECISION,
    dir_acc_ci_low              DOUBLE PRECISION,
    dir_acc_ci_high             DOUBLE PRECISION,
    dir_acc_p_value             DOUBLE PRECISION,
    log_loss_mean               DOUBLE PRECISION,
    base_rate                   DOUBLE PRECISION,

    -- Frozen holdout state
    frozen_holdout_days         INT NOT NULL,
    n_minutes_in_holdout        INT,

    -- Discriminator
    weights_path                TEXT,
    full_report                 JSONB NOT NULL,

    written_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_training_runs_symbol_run_started_at
    ON training_runs (symbol, run_started_at DESC);
```

Trainer's `main()` writes a row at the end of each run (in addition to
the on-disk JSON, which we keep for predictor compat).

**(c) New endpoint: `/api/highfreq/training_history?symbol=...&since_days=7`**

Returns:
```json
{
  "ok": true,
  "symbol": "BTCUSDT",
  "since_days": 7,
  "runs": [
    {"run_started_at": "...", "n_minutes_after_neutral_drop": 1109, "dir_acc_mean": null, ...},
    {"run_started_at": "...", "n_minutes_after_neutral_drop": 1117, "dir_acc_mean": null, ...},
    ...
  ]
}
```

UI: a small sparkline next to "MODEL AGE" showing how `dir_acc_mean`
evolved over the last 7 days. **This** is the defence-grade artefact:
"data accumulated, dir_acc improved monotonically, here's the chart".

### How information flows out of the trainer

```
                          systemd timer fires
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │  python -m app.highfreq.    │
                   │           trainer           │
                   │                             │
                   │  (1) load_seconds(72h)      │
                   │  (2) make_supervised        │
                   │  (3) walk-forward CV        │
                   │  (4) fit final model        │
                   │  (5) write artefacts:       │
                   └──┬──────────────────────────┘
                      │
        ┌─────────────┼─────────────┬─────────────────┐
        ▼             ▼             ▼                 ▼
   .cbm file     metrics.json    Postgres        textfile_collector
   (weights)     (per-symbol     training_runs   heartbeat
                 latest only)    (history)
        │             │              │                │
        ▼             ▼              ▼                ▼
   predictor    /api/highfreq/   /api/highfreq/   Prometheus
   load_model   training_report  training_history alert rules
   on mtime     (current state)  (time series)
   change       + live_inventory
                + live_inventory
                  (real-time
                  bar count)
```

Each artefact is consumed by a different surface — three different
audiences:

| Artefact | Audience | Cadence |
|---|---|---|
| `.cbm` weights | predictor (in-memory model) | hot-reload on mtime |
| `metrics.json` | UI calibration badge, predictor status | overwrite per run |
| `training_runs` row | UI training-history chart, defence | append per run |
| textfile_collector heartbeat | Grafana cron-stale alert | overwrite per run |

---

## 5 · Live trading rollout — 4-stage gate

A no-shortcuts-allowed checklist. **Each stage must be green before
moving to the next.** If any stage regresses, fall back two stages.

```
Stage 1            Stage 2              Stage 3                Stage 4
════════════       ════════════         ════════════           ════════════
SIM ONLY           DEMO MODE            TESTNET LIVE           MAINNET LIVE
(today)            (release C)          (release E?)           (release F?)
                                                               
require_           HF_PAPER_DEMO        HF_EXECUTOR_KIND=      HF_EXECUTOR_KIND=
calibrated=True    _MODE=1              testnet                live
                                                               + HF_LIVE_TRADING
                                                                 _CONFIRMED=1

Trades only fire   Trades fire on raw   Real orders to         Real orders to
when CI > 0.5      P(UP), tagged        testnet.binance.       api.binance.com
                   pre-calibration-     vision (fake $)        (REAL $)
                   demo (excluded
                   from accuracy)
                                                               
Risk: 0            Risk: 0 ($)          Risk: 0 ($) +          Risk: REAL
                                        operational            (capped by
                                        learning               max_qty + halt
                                                               caps)

GATE TO STAGE 2:   GATE TO STAGE 3:     GATE TO STAGE 4:       N/A
- nothing          - frozen holdout     - 30 days clean        (terminal)
                     dir_acc CI low       testnet operation
                     bound > 0.5        - reconciliation
                   - p-value < 0.05       drift = 0
                   - >100 paper         - kill switch
                     trades realized      reachable from
                     accuracy           - independent fee +
                     monotonic            slippage review
```

**Currently we are at Stage 1.** Release C (in flight) ships demo-mode
infrastructure to allow Stage 2 toggle. Stage 3 = wire up the testnet
HTTP/WS client (~1 week of focused work). Stage 4 is a separate moral
+ engineering decision that requires reviewer sign-off, not just code.

---

## 6 · What's in release C (this PR)

* ✅ `app/highfreq/order_executor.py` — Protocol + Sim + stubs.
* ✅ `app/highfreq/data_inventory.py` — real-time fold readiness
  endpoint (already deployed).
* ✅ `paper_trader_runner` — `HF_PAPER_DEMO_MODE` env wire-up.
* ✅ `realized_accuracy` — filters demo trades from skill stats.
* ✅ Tests for all of the above.
* ⏳ Will land in this commit + push.

## 7 · What's in release D (next, after C lands)

Persistent trainer history (section 4(b) and 4(c) above):
* Migration `004_training_runs.sql`.
* Trainer writes a row on every run.
* New endpoint `/api/highfreq/training_history`.
* UI: time-series sparkline next to MODEL AGE.
* Trainer cadence flip: daily → every 4h via env override.

ETA: ~half day's work. Lands as a single coherent PR.

## 8 · What's in release E (when calibration gates clear)

Binance Spot Testnet integration:
* Fill in `BinanceTestnetExecutor.open_long / open_short / close_position`.
* HMAC-SHA256 signed REST client.
* WebSocket user-data stream subscription + listenKey refresh loop.
* `tools/reconcile_orders.py` startup-time consistency check between
  local `paper_trades` and exchange `myTrades`.
* `kill-switch` Telegram bot command (forces all positions to close
  + sets HF_PAPER_HALT_ALL=1 in the env file).
* `tools/dr_drill_executor.py` — verifies the testnet executor can
  open + close a tiny position without losing state on disconnect.

ETA: ~1 week of focused work.

## 9 · What's in release F (mainnet — IF everything else looks great)

Just flip env vars. **No new code.** That's the point of doing the
hard work in stages 2-3 — by the time we'd consider mainnet, the
code path has been exercised for weeks against real Binance API
endpoints (just on testnet money). The mainnet flip is a config
change with documentation, not engineering.

---

## 10 · Open questions for the operator

* **Does the academic defence date fall before Stage 3 (testnet) is
  realistic to ship?** If yes, Stage 2 (demo mode) is the operational
  ceiling for the defence — and that's fine: it's defensible
  ("I built the live-execution scaffold and ran the system end-to-end
  on testnet fixtures, but kept production sim-only until the
  walk-forward CV showed CI > 0.5"). If no, plan ~1 week for Stage 3.
* **Short side.** Binance Spot is long-only. Are we OK keeping shorts
  paper-only forever, or do we want to integrate USDM Futures Testnet
  for proper short execution? Current paper math models a synthetic
  short — fine for demonstration, not fine for live.
* **Trainer cadence final answer.** Default proposal: every 4h.
  Operator can override via env. Sound?
