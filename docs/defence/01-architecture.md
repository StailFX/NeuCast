# §01 — System architecture

> What we built and where each piece lives. This is intentionally a
> **deployable** system, not a notebook — every component has a
> systemd unit, a heartbeat metric, and a recovery runbook.

## 1.1 Two-VPS topology

```
                                 Internet
                                    │
                          https://neucast.ru
                                    │
            ┌──────── Finland VPS (public-facing) ─────────┐
            │  151.245.139.21   user: stailfx              │
            │  • nginx + TLS  • uvicorn (Daily app)        │
            │  • celery + redis  • Postgres (Daily side)   │
            │  • /opt/neucast (source-of-truth for Daily)  │
            │  • Reverse-proxies /highfreq + /grafana to   │
            │    Tokyo via the WireGuard tunnel            │
            └──────────────────┬───────────────────────────┘
                               │
              WireGuard tunnel (UDP/51820, ChaCha20)
              Finland 10.99.0.2 ↔ Tokyo 10.99.0.1
                               │
            ┌────────── Tokyo VPS (HFT slice) ──────────────┐
            │  147.45.49.40    user: root                   │
            │  • Slim FastAPI (HF routes only on 10.99.0.1) │
            │  • Binance L2 ingest (3 symbols, ≈19 ms RTT)  │
            │  • 3× paper-trader runners (1m bars)          │
            │  • 3× CatBoost trainers (systemd timer)       │
            │  • 1× joint multi-symbol trainer (2026-05-09) │
            │  • 3× multi-horizon trainers (5/15/60m)       │
            │  • Postgres 15 (HFT data plane, dockerised)   │
            │  • Prometheus + Grafana + node_exporter       │
            └───────────────────────────────────────────────┘
                               │
                      Yandex Cloud Object Storage
                                    │
                      • L2 cold archive (retention 2d hot)
                      • OFI 1s archive (retention 7d hot)
                      • paper_trades nightly backup
                      • Prometheus TSDB nightly backup
                      • Emergency snapshots
```

## 1.2 Why two VPS

* **Finland** is in Europe — low-latency for the user-facing webapp
  (defence committee will browse it, MSK/SPb users will use it).
  Has the public IP. Runs nginx with Basic Auth on the operator routes.
* **Tokyo** is 19 ms from Binance Spot WebSocket (Tokyo AWS region).
  The whole point of HF is being close to the data. We deliberately
  don't try to do HF from Finland (200+ ms RTT to Binance kills
  edge before the model even gets to predict).
* **WireGuard** carries `/api/highfreq/*` traffic from Finland's
  nginx to Tokyo's slim FastAPI on the private tunnel. From the
  user's browser it looks like one origin (`neucast.ru`); the dual-VPS
  is an implementation detail.

## 1.3 Data plane

```
    Binance Spot WebSocket
            │
            ▼
    L2 ingestor (asyncio, 3 symbols, ≈59 rows/min/symbol)
            │
            ▼
    highfreq_l2_snapshots (Postgres, JSONB book + microprice)
            │
       ┌────┴─────────┐
       ▼              ▼
    (~)4h archive    OFI 1s aggregation (every second)
    to S3            │
                     ▼
              highfreq_ofi_1s (Postgres, ~20 cols microstructure)
                     │
                ┌────┴─────────────┐
                ▼                  ▼
        feature_pipeline    paper_trader_runner
                │           (per-minute tick)
                ▼
        CatBoost trainer
        (per symbol × horizon, daily 04:00 UTC)
                │
                ▼
        weights/highfreq/{btc,eth,bnb}_{1,5,15,60}m.cbm
                │
                ▼
        live LivePredictor (per-process cache)
                │
                ▼
        /api/highfreq/* JSON endpoints
                │
                ▼
        Frontend SPA (Next.js 16, /v2/*)
```

## 1.4 Model artefacts (live as of 2026-05-09)

```
weights/highfreq/
├── btcusdt_1m.cbm + _calibrator.pkl + _metrics.json
├── ethusdt_1m.cbm + _calibrator.pkl + _metrics.json
├── bnbusdt_1m.cbm + _calibrator.pkl + _metrics.json
├── btcusdt_5m.cbm + _metrics.json   ← multi-horizon (long_horizon)
├── ethusdt_5m.cbm + _metrics.json
├── bnbusdt_5m.cbm + _metrics.json
├── btcusdt_15m.cbm + _metrics.json
├── ethusdt_15m.cbm + _metrics.json
├── bnbusdt_15m.cbm + _metrics.json
├── btcusdt_60m.cbm  ← exists but training failed (n_folds=0)
├── joint_1m.cbm + _calibrator.pkl + _metrics.json   ← Phase 2.1, 2026-05-09
├── *_drift.json   ← KS-based drift snapshot per symbol (7d ref vs 6h recent)
├── *_holdout.json ← frozen holdout eval (Mon 04:30 UTC)
├── *_robustness.json ← block-bootstrap robustness suite
├── multi_horizon_eval.json + multi_horizon_compare_features.json
├── joint_multi_horizon.json + joint_60m.json
└── futures_basis_eval.json (cross-venue Spot vs USDM Futures)
```

Total ~30 JSON / .cbm artefacts; every production-relevant one has
a corresponding API endpoint, a section in the dashboard, and is
backed up to S3 nightly.

## 1.5 Cron schedule (Tokyo)

| time UTC | unit | what |
|----------|------|------|
| 02:30 | `neucast-paper-trades-backup.timer` | paper_trades → S3 (no delete) |
| 02:45 | `neucast-ofi-archive.timer` | OFI 1s → S3, retention=7d |
| 03:30 | `neucast-prom-backup.timer` | Prometheus TSDB → S3 |
| 04:00 | `neucast-highfreq-trainer@*.timer` × 3 | per-symbol 1m solo trainers |
| 04:30 (Mon) | `neucast-highfreq-holdout-eval@*.timer` × 3 | frozen holdout |
| 04:45 | `neucast-highfreq-trainer-multihorizon@*.timer` | 5/15/60m trainers |
| **04:50** | **`neucast-highfreq-trainer-joint.timer`** | **joint multi-symbol** |
| every 4h | `neucast-l2-archive.timer` | L2 → S3, retention=2d |
| always-on | `neucast-highfreq.service` | Binance L2 ingest |
| always-on | `neucast-highfreq-web.service` | slim FastAPI on 10.99.0.1 |
| always-on | `neucast-paper-trader@*.service` × 3 | paper-trader runners |

All onesShot crons emit a Prometheus heartbeat to
`/var/lib/prometheus/node-exporter/neucast_hf_<job>_*.prom`. A
single Grafana alert rule (`l2-archive-stale-rule`-style) per heartbeat
fires at 25 h staleness — caught the L2-archive memory-pressure
outage of 2026-05-08 (see `docs/defence/05-engineering-depth.md`).

## 1.6 Frontend stack (release 2026-05-04 → 2026-05-08)

* **Next.js 16** + React 19 + TypeScript + Tailwind CSS 4
* App Router with **static export** (`output: "export"`), basePath
  `/v2`, served by Finland nginx
* TanStack Query for polling + dedup + caching
* `AuthProvider` + cookie-based auth, JSON `/api/auth/{login,register,me,logout}`
* 17 components covering the operator + public dashboards
* `/v2/predict/` form + `/v2/predict/waiting/?task_id=…` polling page
  for the Daily TCN forecast workflow (legacy result HTML for now)
* **55 Vitest tests** (RTL + jsdom + MSW) covering format utils,
  API client (URL construction + ApiError), components (Skeleton,
  DriftBadge, ForecastCard, AuthForm, HorizonPill, Navbar), and
  the predict form workflow

## 1.7 Defence-grade properties this gets you

| concern | answer |
|---------|--------|
| Provenance | Every artefact has a systemd unit, a runbook, an alert. |
| Reproducibility | Every number cited in §02–§04 has a one-line CLI command on Tokyo. |
| Resilience | Two VPS (one can fail). S3 cold archive (Postgres can fail). Heartbeats catch silent crons. |
| Latency budget | 19 ms ingest + ~50 ms model + ~200 ms WireGuard = sub-second user request. |
| Source of truth | GitHub `StailFX/NeuCast`. Tokyo is **deploy target** (rsync), not a git checkout. |
