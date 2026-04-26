# NeuCast High-Frequency Module

A directional 1-minute forecaster for crypto perpetuals (BTCUSDT / ETHUSDT /
BNBUSDT) running end-to-end on a Tokyo-co-located VPS:

```
Binance Spot WS  →  L2 features  →  CatBoost predictor  →  Paper trader
   (~19 ms RTT)      (1 Hz to PG)    (mtime hot-reload)    (sim-only, ADR-005)
                          ↓
                   Live UI / Heatmap / Calibration plot / Telegram alerts
```

**Status:** Phase B + C + D operational · sim-only by [ADR-005](architecture.md#adr-005--sim-backtest-reports-both-maker-and-taker-pl) · multi-symbol (BTC/ETH/BNB) · production-grade observability (Prometheus + Grafana + Yandex S3 cold storage).

**Live URLs:**
* [`neucast.ru/highfreq`](https://neucast.ru/highfreq) — business dashboard (live microprice, orderbook heatmap, predictor status, paper trader log, calibration plot, feature importance)
* [`neucast.ru/grafana`](https://neucast.ru/grafana) — operations dashboard (ingest health, predictor latency, system metrics, Telegram alerts)

> **Why this matters for the portfolio.** The daily-prediction side of NeuCast
> is a yfinance ensemble — interesting, but unable to break the ~1 % MAPE wall
> every OHLCV-only model hits. This module pivots to *microstructure data*
> (order-flow imbalance, microprice, depth ratio), a horizon where the data
> actually carries directional signal, and an evaluation methodology
> (walk-forward CV with bootstrap CI on `dir_acc`) honest enough that a
> "no skill" outcome is *visible* rather than buried.

---

## 1 · Production architecture

```
                                Binance Spot WS (AWS Tokyo, ap-northeast-1)
                                              │ ~19 ms TCP RTT
                                              ▼
   ┌──────────────────── Tokyo VPS (4VPS.su JP-cx21, Ubuntu 24.04) ─────────┐
   │                                                                          │
   │  L2 ingest (asyncio, 1 process, 3 symbols multiplexed)                  │
   │     ├─→ aggregator → highfreq_ofi_1s (1 row/sec/symbol)                 │
   │     └─→ L2SnapshotWriter → highfreq_l2_snapshots (top-10 × 1 Hz)        │
   │                                                                          │
   │  paper-traders × 3 instances (templated systemd, one per symbol)         │
   │     ├─→ predictor (CatBoost) — minute-tick, mtime hot-reload             │
   │     └─→ paper_trader → paper_trades                                      │
   │                                                                          │
   │  trainer @ 04:00 UTC × 3 instances (templated timer per symbol)          │
   │     → walk-forward CV → .cbm + metrics.json                              │
   │                                                                          │
   │  slim FastAPI [8000] — /highfreq + /api/highfreq/* + /metrics/           │
   │  Prometheus [9099] ← scrapes 7 targets every 15s                        │
   │  Grafana [3000/grafana] — dashboards + alert rules → Telegram           │
   │                                                                          │
   │  All HTTP services bind to WG IP (10.99.0.1) only — public only via    │
   │  the encrypted tunnel from Finland nginx.                                │
   └──────────────────────────────────────────────────────────────────────────┘
                                       ↓
                 WireGuard tunnel (ChaCha20-Poly1305, ADR-010)
                                       ↓
            ┌──────── Finland VPS (Hostkey, EU, 151.245.139.21) ──────┐
            │  nginx + Let's Encrypt TLS                                │
            │  ├── /highfreq*       → reverse-proxy → Tokyo:8000        │
            │  ├── /grafana*        → reverse-proxy → Tokyo:3000        │
            │  │   + nginx Basic Auth + rate-limit on /grafana/login    │
            │  └── /, /charts, …    → Finland uvicorn (legacy daily app)│
            └────────────────────────────────────────────────────────────┘
                                       ↓
                          https://neucast.ru/...

                                       ↑
            Yandex S3 cold storage (on user's grant)
            ├── highfreq_l2/{symbol}/{date}.parquet  (>7 days old)
            └── prometheus_snapshots/{date}.tar.gz   (daily backup)
```

---

## 2 · What's in the module (current)

| Layer | File | Purpose |
|---|---|---|
| **Ingest** | `app/highfreq/l2_consumer.py` | Binance WebSocket reader, asyncio, auto-reconnect |
| | `app/highfreq/aggregator.py` | 1-second OFI/microprice/depth windowing → asyncpg writer |
| | `app/highfreq/l2_snapshot_writer.py` | Sub-sampled top-N snapshot writer (1 Hz × 3 symbols × top-10) |
| | `app/highfreq/runner.py` | Standalone entry: `python -m app.highfreq.runner` |
| **Features** | `app/highfreq/feature_pipeline.py` | Pure transforms: aggregate → target → 14 features → live inference |
| **Modeling** | `app/highfreq/trainer.py` | Walk-forward CatBoost + bootstrap CI + JSON report |
| | `app/highfreq/predictor.py` | LivePredictor with mtime hot-reload + per-symbol cache |
| | `app/highfreq/backtest.py` | Maker/taker fee model, P&L curves, Sharpe, fill sweep |
| **Paper trading** | `app/highfreq/paper_trader.py` | State machine: time-stop, vol-adjusted sizing, 3 risk caps |
| | `app/highfreq/paper_trader_runner.py` | Async loop driving the trader once per minute on bar close |
| **Web layer** | `app/highfreq/web.py` | 7 endpoints: status, health, forecast, paper_trades, microprice_history, orderbook, regimes, feature_importance, training_report |
| | `app/highfreq/web_app.py` | Slim FastAPI — only HFT routes (no TF/Torch import chain) |
| | `templates/highfreq.html` | Live UI with 11+ visualisation blocks (see §4) |
| **Tools** | `app/highfreq/regimes.py` | Vol-regime classifier (rolling percentile) |
| | `app/highfreq/threshold_search.py` | Offline grid-search of entry/exit thresholds |
| | `app/highfreq/metrics.py` | Centralised Prometheus metric registry |
| | `tools/archive_l2_to_s3.py` | Daily atomic archival of L2 snapshots → Yandex S3 |
| | `tools/backup_prometheus_to_s3.py` | Daily Prometheus TSDB snapshot → Yandex S3 |
| **Tests** | `tests/test_highfreq_*.py` | **414 tests, all passing**, no live DB / WS / FastAPI client |

---

## 3 · UI surface — what `/highfreq` shows (top-to-bottom)

1. **Header** — symbol selector dropdown (BTC / ETH / BNB) + phase badge
2. **Vol regime pill** — `calm` / `normal` / `turbulent` + 60-min strip chart
3. **Metrics grid** — microprice / OFI / depth_imb / spread / RPS / age + data-readiness progress bar
4. **Live microprice chart** — last 5 min SVG line, ±bps stat, polls every 2s
5. **Orderbook density heatmap** — Bookmap-style canvas (X=time, Y=price levels, color=log(qty))
6. **Forecast widget** — P(up) + signal arrow + dir_acc CI + model age + walk-forward fold readiness
7. **Paper trading status** — 6 cards: status / today P&L / trades W/L / loss streak / lifetime / last trade
8. **Cumulative P&L sparkline** — SVG line with auto-scale
9. **Recent trades table** — last 20 closed paper trades with side / prices / qty / P&L / reason
10. **Feature importance** — horizontal bars from CatBoost native importance, polls every 30s
11. **Walk-forward calibration plot** — per-fold dir_acc with Wilson 95% CI bars + 0.50 random + base-rate baselines

---

## 4 · Headline architecture decisions (11 ADRs)

Full details in [`architecture.md`](architecture.md):

| ADR | Decision | Why |
|---|---|---|
| 001 | In-memory aggregation, no TimescaleDB | 250 MB/month vs 50 GB raw |
| 002 | Persist Binance event time, not local clock | Future Tokyo migration stays valid |
| 003 | 1-minute horizon, not 1-second | 19 ms RTT = 0.03 % of horizon |
| 004 | Direction loss, not MAPE | MAPE optimum ≠ trading optimum |
| 005 | Sim-only, both maker + taker P&L reported | Honest reporting > cherry-picked |
| 006 | Coexist via resource limits *(superseded by 009)* | Capacity skill = portfolio artefact |
| 007 | Binary classification + neutral-band drop | Removes noise class |
| 008 | Expanding-window walk-forward, not random k-fold | Random k-fold leaks future |
| **009** | **Tokyo VPS as single source of truth for HFT data** | 13× lower latency, single-DB topology |
| **010** | **WireGuard tunnel for Finland↔Tokyo HTTP** | Defence in depth, kernel-fast crypto |
| **011** | **Paper-trading contract** (time-stop, maker fees, halt rules) | One doc to defend the trading semantics |

---

## 5 · Operational stack on Tokyo VPS

8 systemd services + 5 timers, all visible in:
```bash
systemctl list-units 'neucast-*'
systemctl list-timers 'neucast-*'
```

| Service / Timer | Cadence | What |
|---|---|---|
| `neucast-highfreq.service` | always-on | L2 WebSocket ingest |
| `neucast-highfreq-web.service` | always-on | slim FastAPI for /highfreq + /metrics |
| `neucast-paper-trader@{btc,eth,bnb}usdt.service` | always-on | 3 paper-trader runners |
| `neucast-highfreq-trainer@{...}.timer` | 04:00 UTC daily × 3 | CatBoost retraining per symbol (last 7 days held out) |
| `neucast-highfreq-holdout-eval@{...}.timer` | Mon 04:30 UTC × 3 | Frozen-holdout OOS eval — true-OOS dir_acc + p-value |
| `neucast-l2-archive.timer` | every 4h UTC | L2 snapshots > 2 days → Yandex S3, atomic verify-before-delete |
| `neucast-paper-trades-backup.timer` | 02:30 UTC daily | paper_trades → Yandex S3 (backup only, no delete) |
| `neucast-ofi-archive.timer` | 02:45 UTC daily | OFI 1-sec rows > 7 days → Yandex S3, atomic verify-before-delete |
| `neucast-prom-backup.timer` | 03:30 UTC daily | Prometheus TSDB snapshot → Yandex S3 |
| `prometheus.service` | always-on | TSDB + scraper (port 9099, 30 d retention) |
| `grafana-server.service` | always-on | dashboards + alerting |
| `prometheus-node-exporter.service` | always-on | system metrics (CPU/RAM/disk) |

**Health-check one-liner:**
```bash
curl -s -u neucast:<basic_pass> https://neucast.ru/api/highfreq/health
# {"ok": true, "symbol": "BTCUSDT", "rows_last_60s": 60}
```

---

## 6 · Tests

```bash
python3 -m pytest tests/ -q
# 414 passed in ~5 s
```

All 414 tests are pure-function — no Postgres, no FastAPI client, no live
WebSocket. The test pyramid:

| Test file | Tests | Covers |
|---|---:|---|
| `test_highfreq_aggregator.py` | 24 | OFI math, async batch flush, reconnect resync |
| `test_highfreq_l2_consumer.py` | 25 | WS frame parsing, dispatch, reconnect logic |
| `test_highfreq_l2_snapshot_writer.py` | 14 | sub-sampling, top-N trim, batch flush, error resilience |
| `test_highfreq_trainer.py` | 14 | minute aggregation, target shift, CV, bootstrap CI |
| `test_highfreq_backtest.py` | 20 | fees, ledger, Sharpe, fill sweep |
| `test_highfreq_predictor.py` | 24 | hot-reload, calibration gate, feature reorder, per-symbol cache |
| `test_highfreq_feature_pipeline.py` | 11 | aggregation, neutral-band, build_latest_inference_bar |
| `test_highfreq_paper_trader.py` | 58 | all state-machine branches, vol-adjusted sizing, risk caps, day rollover |
| `test_highfreq_paper_trader_runner.py` | 22 | timing helpers, async tick, force-close on shutdown |
| `test_highfreq_paper_trades_endpoint.py` | 26 | 6 endpoints × cold-start / populated / db-unavailable / clamps |
| `test_highfreq_threshold_search.py` | 8 | grid-search re-classification, Sharpe arithmetic |
| `test_highfreq_regimes.py` | 14 | percentile thresholds, label_history, edge cases |
| `test_highfreq_web.py` | 26 | UI payload, sanitisation, edges |
| `test_highfreq_forecast_endpoint.py` | 18 | 4 × 503 branches + happy path |
| `test_archive_l2_to_s3.py` | 13 | atomic verify-before-delete, idempotency, failure paths |
| `test_error_pages.py` | 10 | (root-level error rendering, kept for parity) |

---

## 7 · Daily-rate observability metrics

Exposed via Prometheus on each service (`/metrics` endpoint):

| Metric | Source | Used by |
|---|---|---|
| `neucast_hf_ws_frames_total` | ingest | Grafana panel + "ingest dead" alert |
| `neucast_hf_ws_reconnects_total` | ingest | "reconnect storm" alert |
| `neucast_hf_ofi_rows_written_total{symbol}` | ingest | "OFI stalled" alert |
| `neucast_hf_l2_snapshots_written_total{symbol}` | ingest | dashboard panel |
| `neucast_hf_predictions_total{symbol, signal}` | runner | dashboard panel |
| `neucast_hf_prediction_latency_seconds` | runner | p50/p95/p99 panel |
| `neucast_hf_predictor_calibrated{symbol}` | runner | dashboard gauge |
| `neucast_hf_predictor_dir_acc_ci_low{symbol}` | runner | dashboard panel |
| `neucast_hf_paper_trades_opened_total{symbol, side}` | runner | dashboard panel |
| `neucast_hf_paper_trades_closed_total{symbol, exit_reason}` | runner | dashboard panel |
| `neucast_hf_paper_pnl_usd_total{symbol}` | runner | cumulative P&L panel |
| `neucast_hf_paper_trader_halted{symbol, reason}` | runner | "trader halted" alert |
| `neucast_hf_paper_consecutive_losses{symbol}` | runner | dashboard gauge |
| `node_*` | node-exporter | system panels (CPU/RAM/disk) |

Telegram alerts route via the `telegram-stailfx` contact point with a custom HTML template.

---

## 8 · Storage layout

| Postgres table | Cadence | Retention | Note |
|---|---|---|---|
| `highfreq_ofi_1s` | 1 row/sec/symbol | forever | predictor input + UI metrics |
| `highfreq_features_1m` | populated by trainer | forever | walk-forward CV target |
| `highfreq_l2_snapshots` | 1 snapshot/sec/symbol | **7 days** | older → Yandex S3 |
| `paper_trades` | one per closed trade | forever | post-Tier-2 gate decision |

| Yandex S3 prefix | Cadence | Format |
|---|---|---|
| `highfreq_l2/{symbol}/{date}.parquet` | daily | snappy-compressed Parquet |
| `prometheus_snapshots/{date}.tar.gz` | daily | tar+gz of TSDB snapshot |
| `test-snapshot/` | one-off | proof-of-concept upload |

---

## 9 · Deployment

Production runs on a Tokyo VPS bootstrapped from clean Ubuntu 24.04 in
~5 minutes via [`deploy/bootstrap_tokyo.sh`](deploy/bootstrap_tokyo.sh).
Full ops runbook + all 11 systemd unit files: [`deploy/`](deploy/).

The Finland-side nginx config (TLS termination + reverse-proxy of
`/highfreq*` and `/grafana*` to Tokyo over WireGuard) is documented in
[ADR-009](architecture.md#adr-009--tokyo-vps-as-the-hft-data-plane-supersedes-adr-006-for-the-hft-slice)
and [ADR-010](architecture.md#adr-010--wireguard-tunnel-for-finland↔tokyo-http-traffic).

---

## 10 · Reference

* Architecture + 11 ADRs: [`architecture.md`](architecture.md)
* Demo recipe with sample output: [`demo.md`](demo.md)
* Deployment runbook: [`deploy/README.md`](deploy/README.md)
* WireGuard tunnel setup: [`deploy/wireguard_setup.md`](deploy/wireguard_setup.md)
* SQL DDLs: [`../../app/highfreq/migrations/`](../../app/highfreq/migrations/)

External:

* Cont, Kukanov, Stoikov (2014) — *The Price Impact of Order Book Events* (OFI)
* Stoikov (2018) — *The Micro-Price* (depth-weighted mid)
* López de Prado (2018) — *Advances in Financial Machine Learning*, §7.4 (purged k-fold; ADR-008's Phase B upgrade target)
* Wilson (1927) — *Probable Inference, the Law of Succession, and Statistical Inference* (per-fold CI in calibration plot)
* Moskowitz et al. (2012) — *Time Series Momentum*, §4 (vol-adjusted sizing rationale)
* Binance Spot API: [streams](https://binance-docs.github.io/apidocs/spot/en/#websocket-market-streams) · [fees](https://www.binance.com/en/fee/schedule)
