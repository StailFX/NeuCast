# NeuCast High-Frequency Module — Architecture (Phase A)

**Status:** Phase A — sim-backtest only
**Goal:** 1-minute directional forecast for BTC-USDT, with honest paper-trading simulation
**Owner:** Stailfx (coursework / portfolio project)
**Last update:** 2026-04-25

---

## 1 · Goal

Build a parallel high-frequency forecasting product that lives next to the existing daily-prediction service. The HF module:

- consumes Binance Spot Level-2 order book stream + trade stream via public WebSocket
- engineers Order-Flow-Imbalance (OFI), microprice, depth-imbalance features at 1-second granularity
- trains a CatBoost classifier on `sign(return_1m)` with directional log-loss
- runs walk-forward sim-backtest with realistic Binance fees (maker rebate vs taker cost)
- surfaces live forecast + paper P&L + rolling directional accuracy on a new `/highfreq` UI page

**Phase A explicitly does NOT:**
- place real or paper orders against Binance
- predict at sub-1-minute horizons
- require sub-100ms inference latency

**This document records the design decisions made up-front, with reasoning and trade-offs, so a reader can understand *why* the system is shaped the way it is.**

---

## 2 · Constraints

The module shares a single 8 GB VPS (Hostkey Finland) with three unrelated production services. This shapes every decision below.

| Resource | Total | Currently used | Available for HF |
|----------|-------|----------------|--------------------|
| CPU cores | 4 | ~30 % avg | ≥1 core, bursts to 4 |
| RAM | 7.7 GB | 5.5 GB | 1.9 GB + 4 GB swap (added) |
| Disk | 118 GB | 54 GB | 60 GB free |
| Network → Binance | — | — | ~150 ms RTT (Finland → Asia) |

**Coexistence requirement:** the HF module must not destabilize the daily NeuCast service or the unrelated apps on the same host. This is enforced via Docker memory limits, single-thread CatBoost training, and scheduling heavy work in low-traffic UTC hours.

---

## 3 · Architecture diagram

```
┌──────────────────┐      WebSocket          ┌─────────────────────┐
│ Binance Spot WSS │ ───────────────────────▶│   L2 Consumer       │
│  depth20@100ms   │   ~150 ms RTT, ~5 MB/s  │   (asyncio task)    │
│  trade stream    │                         └──────────┬──────────┘
└──────────────────┘                                    │
                                                        │ in-memory ring buffer
                                                        │ (1 s window, ~50 MB)
                                                        ▼
                                            ┌─────────────────────┐
                                            │  OFI Aggregator     │
                                            │  1-second features  │
                                            └──────────┬──────────┘
                                                       │ ~100 B/sec
                                                       ▼
                                            ┌─────────────────────┐
                                            │  Postgres           │
                                            │  ofi_features_1s    │
                                            │  ofi_features_1m    │
                                            └──────────┬──────────┘
                                                       │
                          ┌────────────────────────────┼─────────────────────────────┐
                          ▼                            ▼                             ▼
                ┌──────────────────┐       ┌──────────────────────┐      ┌────────────────────┐
                │ CatBoost trainer │       │  Sim-backtest engine │      │  Live predictor    │
                │ (Celery beat,    │       │  walk-forward,       │      │  (FastAPI route)   │
                │  04:00 UTC)      │       │  maker/taker fees    │      │  /highfreq/latest  │
                └──────────────────┘       └──────────────────────┘      └────────────────────┘
                                                       │                             │
                                                       └──────────────┬──────────────┘
                                                                      ▼
                                                          ┌─────────────────────┐
                                                          │  /highfreq UI       │
                                                          │  Live + backtest    │
                                                          │  charts             │
                                                          └─────────────────────┘
```

---

## 4 · Architecture Decision Records

Each ADR captures one non-obvious decision, the alternatives considered, and the trade-off accepted.

### ADR-001 · In-memory aggregation, no TimescaleDB

**Context.** Storing raw L2 depth-20 snapshots at 100 ms cadence costs ~50 GB / 3 months. The deployed Postgres image is `postgres:15-alpine`, which does not bundle the TimescaleDB extension; switching the image requires a 30-minute downtime migration and ~50 GB of additional disk.

**Decision.** Aggregate the L2 stream in process memory into 1-second OFI features and persist *only* the aggregates. Raw snapshots are kept for 24 hours rolling in an on-disk SQLite buffer for live debugging, then dropped.

**Why this works.**
- 1-second OFI features → 1 row × ~100 bytes × 86 400 s/day = ~8 MB/day = ~250 MB/month. Fits the 60 GB disk headroom indefinitely.
- 1-minute model never queries sub-second history beyond a 60-row rolling window held in memory.
- TimescaleDB compression would deliver the same ~95 % footprint reduction *after* ingest; doing it at ingest is cheaper.

**Trade-off accepted.** We cannot retrospectively re-extract sub-second features. If we later decide we want a 1-second prediction horizon (Phase B), we have to design the feature in advance and replay the live stream forward — we cannot mine the historical raw L2.

**Alternative rejected.** Deploying TimescaleDB. Reason: storage savings are real but small (~30 GB compressed vs ~250 MB aggregated); the migration cost is non-trivial; and the operational surface increases (new extension, new compaction policies).

---

### ADR-002 · Use Binance event time, not local timestamp

**Context.** WebSocket frames take ~150 ms to reach our Finland VPS from Binance's Asia datacenter. If we used `time.time()` at receive, our timestamps would lag market reality by a variable amount (~50–250 ms with jitter), making the dataset incompatible with any future migration to a Tokyo VPS.

**Decision.** Persist the `E` field (event time, UTC ms since epoch) from each WebSocket frame as the canonical timestamp. Local-receive time is also recorded but only as a *diagnostic* column (for monitoring network jitter), never used by ML.

**Why this works.**
- `E` is set by Binance's matching engine and is the same regardless of where we listen from.
- Sequence ordering is preserved across servers — datasets collected on Finland VPS remain valid if we later move ingest to Tokyo.
- Live inference and historical training can share a code path with no timezone drift.

**Trade-off accepted.** We trust Binance's clock. They have published SLAs on event-time accuracy (< 5 ms drift), which is well below our 1-minute horizon.

---

### ADR-003 · 1-minute forecast horizon, not 1-second

**Context.** True HFT shops predict at 1-100 ms horizons and require co-location. We predict from a 150-ms-latency Finland VPS.

**Decision.** Phase A targets `direction(t + 60 s)`, not `direction(t + 1 s)`.

**Why this works.**
- 150 ms latency = 0.25 % of a 60-second window — negligible.
- 150 ms latency = 15 % of a 1-second window — would invalidate sub-second predictions.
- 1-minute features (rolling 1 s OFI over 60 s) have substantially less noise than tick-level features.
- Binance publishes 1-minute klines historically, so we can sanity-check the OFI-driven model against simpler baselines (returns-only).

**Trade-off accepted.** Theoretical edge ceiling is lower than a 1-second predictor, but 1-min is realistic for our infrastructure. The Phase A → Phase B migration path explicitly addresses 1-s prediction with a Tokyo VPS if and when Phase A succeeds.

---

### ADR-004 · Direction loss, not MAPE

**Context.** The daily NeuCast ensemble has been trained on MSE / MAPE losses for months and has converged to MAPE ≈ 1.85 % on BTC-USDT — which is the known physical floor for OHLCV-only forecasting (Foundation models like Chronos / TimesFM hit the same wall, see [Chronos paper](https://arxiv.org/abs/2403.07815)). MAPE is *not* a useful trading metric: a "tomorrow ≈ today" predictor scores 1.85 % MAPE while making zero PnL.

**Decision.** The Phase A model is a CatBoost **classifier** on `sign(return_1m)`, optimizing log-loss with class weights to handle directional imbalance.

**Why this works.**
- Loss aligns with the metric we report (directional accuracy + paper P&L).
- Signed targets sidestep the price-level convergence trap that punished the daily model.
- CatBoost's GPU-free training fits the resource budget.

**Trade-off accepted.** We lose return-magnitude information. For Phase B we may add a regression head to size positions, but Phase A is direction-only.

---

### ADR-005 · Sim-backtest reports both maker and taker P&L

**Context.** Binance fees fundamentally change strategy economics:

| Side | Fee | Effect on strategy with 54 % dir_acc |
|------|-----|--------------------------------------|
| Taker (market order) | 0.1 % per trade | Net negative — fee eats edge |
| Maker (post-only limit) | **−0.001 %** (rebate) | Net positive — exchange pays you |

A backtest reporting only one of these is misleading. A backtest reporting only taker P&L would tell us the strategy is unprofitable when actually it is profitable as a maker. A backtest reporting only maker P&L would over-estimate, because maker fill rates are uncertain (~30–50 % in our regime).

**Decision.** The backtest engine produces *both* P&L curves on every run, plus an explicit fill-rate sensitivity sweep (assume 30 / 50 / 70 / 100 % maker fill).

**Why this works.**
- Honest reporting is the central value of this project.
- Reader of the portfolio writeup sees the full economic picture, not a cherry-pick.
- Forces us to articulate what fill rate we believe in, rather than assuming it.

**Trade-off accepted.** More numbers to communicate. We address this by always reporting maker / taker side by side and summarising in a single "deployable as maker only" badge on the UI.

---

### ADR-006 · Coexist via resource limits, not isolation

**Context.** We could pay ~900 ₽/month for a second VPS to isolate the HF module. We chose not to.

**Decision.** Run HF module on the existing Hostkey Finland VPS, alongside daily-NeuCast / gymbro / vita-balance, with explicit resource limits:

- Docker `mem_limit: 1.5g` for the L2 consumer container
- CatBoost `thread_count = 2`, training scheduled at 04:00 UTC (low-traffic window for the other apps)
- 4 GB swap added as OOM-killer safety net (already provisioned)

**Why this works for a coursework / portfolio project.**
- Zero additional infrastructure cost.
- Demonstrates capacity-planning skill — explicit limits + monitoring is *itself* a portfolio artifact.
- Acceptable risk: no live capital is at stake in Phase A; a brief slowdown is recoverable.

**Trade-off accepted.** If gymbro spikes hard, our pipeline may briefly slow or hit swap. We monitor this and document it. A real production HF system would not run on shared infrastructure — this is an explicit Phase A scope decision, not a permanent design.

---

## 5 · Module layout

```
app/
└── highfreq/
    ├── __init__.py
    ├── l2_consumer.py        # WebSocket → in-memory ring buffer
    ├── ofi_features.py       # OFI / microprice / depth-imbalance computation
    ├── aggregator.py         # 1-s and 1-m feature aggregation, Postgres writer
    ├── trainer.py            # CatBoost walk-forward training (Celery task)
    ├── predictor.py          # Live inference + FastAPI route
    ├── backtest.py           # Sim-backtest engine, maker / taker fee model
    └── paper_trading.py      # Optional paper-PnL tracker (Phase A.5)

docs/
└── highfreq/
    ├── architecture.md       # this file
    ├── adr-NNN-*.md          # individual ADRs (linked from §4 above)
    └── results/              # backtest reports, screenshots — populated as we go

templates/
└── highfreq.html             # /highfreq UI page

scripts/
└── highfreq_health.py        # operator script: WebSocket alive? ticks/min? last forecast?
```

---

## 6 · Data schema

Two tables, both in the existing `neucast` database (port 5433):

```sql
-- 1-second OFI features (~250 MB/month, retained indefinitely)
CREATE TABLE highfreq_ofi_1s (
    ts            TIMESTAMPTZ NOT NULL,    -- Binance event time, see ADR-002
    symbol        TEXT        NOT NULL,    -- e.g. 'BTCUSDT'
    ofi           DOUBLE PRECISION,        -- order-flow imbalance
    microprice    DOUBLE PRECISION,        -- depth-weighted mid
    depth_imb     DOUBLE PRECISION,        -- top-N bid vs ask depth ratio
    spread_bps    DOUBLE PRECISION,        -- (ask - bid) / mid * 10000
    trade_imb     DOUBLE PRECISION,        -- aggressive buy vs sell volume
    vpin          DOUBLE PRECISION,        -- volume-bucket informed-trade probability
    n_updates     INTEGER,                 -- number of L2 updates in this 1-s window
    local_recv_ms INTEGER,                 -- local jitter diagnostic, NOT for ML
    PRIMARY KEY (ts, symbol)
);
CREATE INDEX ON highfreq_ofi_1s (symbol, ts DESC);

-- 1-minute aggregated features for model training
CREATE TABLE highfreq_features_1m (
    ts            TIMESTAMPTZ NOT NULL,    -- minute boundary in event time
    symbol        TEXT        NOT NULL,
    -- aggregates of 1-s features over the minute:
    ofi_mean      DOUBLE PRECISION,
    ofi_sum       DOUBLE PRECISION,
    ofi_std       DOUBLE PRECISION,
    microprice_open   DOUBLE PRECISION,
    microprice_close  DOUBLE PRECISION,
    microprice_mean   DOUBLE PRECISION,
    depth_imb_mean    DOUBLE PRECISION,
    spread_bps_mean   DOUBLE PRECISION,
    trade_imb_sum     DOUBLE PRECISION,
    vpin_mean         DOUBLE PRECISION,
    -- target (filled in after t+1m completes):
    return_1m         DOUBLE PRECISION,
    direction         SMALLINT,             -- sign(return_1m): -1, 0, +1
    PRIMARY KEY (ts, symbol)
);
CREATE INDEX ON highfreq_features_1m (symbol, ts DESC);
```

Both tables coexist with the existing daily-prediction tables — no conflicts.

---

## 7 · Roadmap

| Phase | Effort | Outcome |
|-------|--------|---------|
| **A.0 · Setup** ✅ | 1 day | swap added, dirs, this doc |
| A.1 · Architecture polish | 1 day | this doc finalised, ADR cross-links |
| A.2 · L2 consumer | 2 days | WebSocket alive, ticks landing in Postgres |
| A.3 · OFI features | 1 day | feature columns populated correctly |
| A.4 · CatBoost trainer | 2 days | model trained, reports dir_acc |
| A.5 · Sim-backtest | 2 days | maker / taker P&L curves, fill-rate sweep |
| A.6 · UI page | 1 day | `/highfreq` shows live forecast + backtest |
| A.7 · Polish + README | 1 day | portfolio-ready repo |

**Total: ~11 working days of implementation. Calendar: ~3 weeks at 5–10 h/week of user-side validation.**

**Gate to Phase B.** If after 2 weeks of live data collection the sim-backtest shows directional accuracy ≥ 53 % on out-of-sample data with a positive maker P&L curve, we proceed to Phase B (1-second predictions, paper trading on live WebSocket). If not, we close the project as "negative result, documented" — which is *also* a valid portfolio outcome.

---

## 8 · Out of scope for Phase A

To prevent scope creep, these are explicitly *not* part of Phase A:

- Real-money trading
- Sub-1-minute prediction horizons
- Multi-asset basket / cross-sectional ranking
- LLM-based news / sentiment fusion
- Tokyo VPS migration
- Reinforcement-learning-based execution

Each of the above has an obvious slot in a Phase B or C plan and will be addressed there.

---

*This document evolves with the project. Each non-obvious change must add an ADR; small changes can amend §3–§6 in place.*
