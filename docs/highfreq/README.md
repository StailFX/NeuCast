# NeuCast High-Frequency Module — Phase A

A 1-minute directional forecaster for BTC-USDT, sitting next to the
existing daily-prediction service on a single 8 GB VPS. Built end-to-end:
real-time L2 ingest → microstructure features → walk-forward CatBoost →
sim-backtest with realistic Binance fees → live UI.

**Status:** Phase A complete · sim-backtest only · no live orders ever
(see [ADR-005](architecture.md#adr-005--sim-backtest-reports-both-maker-and-taker-pl)).

> **Why this matters for the portfolio.** The daily-prediction side of NeuCast
> is a Yahoo-Finance ensemble — interesting, but unable to break the
> ~1 % MAPE wall that every OHLCV-only model hits. This module pivots
> to *microstructure data Yahoo doesn't publish* (order-flow imbalance,
> microprice, depth ratio), a horizon where the data actually carries
> directional signal, and an evaluation methodology (walk-forward with
> bootstrap CI on `dir_acc`) honest enough that a "no skill" outcome is
> visible rather than hidden.

---

## 1 · Architecture at a glance

```
Binance Spot WSS ──▶  L2 Consumer (asyncio, 1 task)
 depth20@100ms        ├─ event time, no local-clock drift  (ADR-002)
 trade stream         └─ in-memory ring buffer, 1-s window
                              │
                              ▼  ~100 B/sec aggregated
                      OFI Aggregator
                       1-second features  ──▶  Postgres (~250 MB/month)
                              │                 highfreq_ofi_1s
                              ▼                 highfreq_features_1m
                      ┌───────┴────────────────┬─────────────────────┐
                      ▼                        ▼                     ▼
              CatBoost trainer       Sim-backtest engine        FastAPI router
              walk-forward CV        maker/taker P&L            /highfreq UI
              bootstrap CI           fill-rate sweep            /api/highfreq/*
```

Full diagram + 8 architecture decisions: [`architecture.md`](architecture.md).

---

## 2 · What's in the module

| File | LOC | Role |
|------|----:|------|
| `app/highfreq/l2_consumer.py` | 266 | Binance WebSocket reader, asyncio, auto-reconnect |
| `app/highfreq/ofi_features.py` | 129 | OFI / microprice / depth-imbalance — pure NumPy |
| `app/highfreq/aggregator.py` | 274 | 1-second windowing, asyncpg batch writer |
| `app/highfreq/runner.py` | 148 | Standalone entry: `python -m app.highfreq.runner` |
| `app/highfreq/trainer.py` | 729 | Walk-forward CatBoost + bootstrap CI + JSON report |
| `app/highfreq/backtest.py` | 410 | Maker/taker fee model, P&L curves, Sharpe, fill sweep |
| `app/highfreq/web.py` | 390 | FastAPI router, status/health/HTML endpoints |
| `templates/highfreq.html` | 442 | Live UI: microprice, OFI, countdown bar |
| **Source total** | **~2 800** | |
| `tests/test_highfreq_trainer.py` | 196 | 14 tests — feature pipeline, CV, bootstrap CI |
| `tests/test_highfreq_backtest.py` | 298 | 20 tests — fees, ledger, Sharpe, sweep, JSON |
| `tests/test_highfreq_web.py` | 286 | 26 tests — UI payload, sanitisation, edges |
| **Test total** | **~780 LOC, 60 tests, 0.7 s** | |
| `docs/highfreq/architecture.md` | 342 | This document's design-decision sibling |

Pre-existing daily-prediction code (TCN ensemble, Foundation models) is
left untouched.

---

## 3 · Headline design decisions

| ADR | Decision | Why |
|---|---|---|
| [001](architecture.md#adr-001--in-memory-aggregation-no-timescaledb) | Aggregate L2 in process memory; persist 1-s features only | 250 MB/month vs 50 GB raw, no TimescaleDB extension needed |
| [002](architecture.md#adr-002--use-binance-event-time-not-local-timestamp) | Persist Binance event time, not local `time.time()` | Future Tokyo VPS migration stays valid |
| [003](architecture.md#adr-003--1-minute-forecast-horizon-not-1-second) | Forecast `direction(t+60s)`, not `direction(t+1s)` | 150 ms RTT = 0.25 % of horizon (negligible) vs 15 % at 1-s |
| [004](architecture.md#adr-004--direction-loss-not-mape) | Classify `sign(return_1m)`, not regress price | MAPE optimum ≠ trading optimum (the daily-side lesson) |
| [005](architecture.md#adr-005--sim-backtest-reports-both-maker-and-taker-pl) | Report **both** maker and taker P&L, every run | Honest reporting > cherry-picked single number |
| [006](architecture.md#adr-006--coexist-via-resource-limits-not-isolation) | Coexist on shared VPS with explicit memory limits | Capacity-planning skill is itself a portfolio artifact |
| [007](architecture.md#adr-007--binary-classification-with-neutral-band-drop) | Drop bars with `\|return\|<1bp` before training | Removes degenerate "noise class" |
| [008](architecture.md#adr-008--expanding-window-walk-forward-not-random-k-fold) | Expanding-window walk-forward CV, not random k-fold | Random k-fold leaks future → inflates `dir_acc` 3-5 pp |

---

## 4 · Demo loop

End-to-end walkthrough lives in [`demo.md`](demo.md). TL;DR:

```bash
# 1. Start the L2 ingest service (locally or on VPS)
python -m app.highfreq.runner

# 2. Watch progress on the UI
open http://localhost:8100/highfreq

# 3. After ≥65 minutes accumulated, train the first model
python -m app.highfreq.trainer \
    --symbol BTCUSDT --since-hours 24 \
    --out weights/highfreq/btcusdt_$(date +%Y%m%d).cbm

# 4. Run the sim-backtest on the trained predictions
python -m app.highfreq.backtest \
    --predictions weights/highfreq/btcusdt_$(date +%Y%m%d)_predictions.parquet \
    --out reports/highfreq/btcusdt_$(date +%Y%m%d).json
```

---

## 5 · Running the tests

```bash
python3 -m pytest tests/ -v
```

Expected:

```
============================== 60 passed in 0.68s ==============================
```

All 60 tests are pure-function — no Postgres, no FastAPI client, no live
WebSocket. They cover:

* **Trainer (14):** minute aggregation, target shift, neutral-band flagging,
  feature schema invariance, bootstrap CI determinism + edge cases.
* **Backtest (20):** fee endpoints + clamping, long/short ledger PnL,
  threshold filtering, chronological cumsum invariant, max-drawdown,
  Sharpe annualisation, perfectly-predictive vs anti-predictive sims,
  fill-rate sweep monotonicity, RFC-7159 JSON output.
* **Web (26):** NaN/Inf scrubbing, defensive `_to_float`, progress-bar
  clamping, naive-timestamp coercion, age clock-skew clamping, status
  assembly across no-data / fresh / stale / enough-data paths.

---

## 6 · Deployment

Production runs on a Hostkey Finland VPS as a systemd unit alongside the
existing `neucast.service` (uvicorn) and `neucast-celery.service`. Full
ops runbook + sanitised systemd unit: [`deploy/`](deploy/).

Health-check pattern:

```bash
ssh vps 'sudo systemctl status neucast-highfreq.service --no-pager'
ssh vps 'sudo journalctl -u neucast-highfreq.service --since "5 minutes ago"'

curl https://neucast.ru/api/highfreq/health
# {"ok": true, "symbol": "BTCUSDT", "rows_last_60s": 60}
```

---

## 7 · What's next

* **Phase A.2.2 (passive watch).** 24-48 h soak — reconnects, RAM drift,
  growth-rate of `highfreq_ofi_1s`. Pure operations, no code change.
* **Phase A.7 polish (this).** README + demo + screenshot placeholders.
* **Phase B (gated).** Triggered only if 2 weeks of accumulated data
  show `dir_acc ≥ 53 %` with positive maker P&L. Sub-1-minute horizon,
  Tokyo VPS, paper trading on live WebSocket. See architecture §7.

A "negative result" outcome (no skill detected) is documented in
architecture §7 as an explicitly valid Phase A end-state — the
methodology is the deliverable, not the prediction.

---

## 8 · Reference

* Architecture + ADRs: [`architecture.md`](architecture.md)
* Demo recipe with sample output: [`demo.md`](demo.md)
* Deployment runbook: [`deploy/README.md`](deploy/README.md)
* SQL DDL: [`../../app/highfreq/migrations/001_initial_schema.sql`](../../app/highfreq/migrations/001_initial_schema.sql)

External:

* Cont, Kukanov, Stoikov (2014) — *The Price Impact of Order Book Events* (OFI)
* Stoikov (2018) — *The Micro-Price* (depth-weighted mid)
* López de Prado (2018) — *Advances in Financial Machine Learning*, §7.4 (purged k-fold; ADR-008's Phase B upgrade target)
* Binance Spot API: [streams](https://binance-docs.github.io/apidocs/spot/en/#websocket-market-streams) · [fees](https://www.binance.com/en/fee/schedule)
