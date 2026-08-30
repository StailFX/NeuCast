# Phase A — Deployment notes

This directory contains deploy artifacts referenced by
[`../architecture.md`](../architecture.md). Three deployment paths exist; pick
based on where you're starting from.

| Mode | Where | When |
|---|---|---|
| **Tokyo greenfield (ADR-009)** | `bootstrap_tokyo.sh` | Fresh Ubuntu 22.04/24.04 VPS in Tokyo — single command from clean OS to running ingest + slim web. **This is the current production path.** |
| Docker Compose | `docker-compose.yml` → `highfreq-l2` service | Local dev only |
| Systemd (legacy) | `neucast-highfreq.service` (this dir) | Existing Hostkey Finland VPS where app/celery already run as systemd units. Per ADR-009, ingest no longer runs here — kept for reference. |

## Production layout (post ADR-009 / ADR-010 / monitoring)

```
TOKYO 147.45.49.40 (4VPS.su JP-cx21, Ubuntu 24.04)    FINLAND 151.245.139.21 (Hostkey)
─ neucast-highfreq.service       (L2 ingest)          ─ nginx + Let's Encrypt TLS
─ neucast-highfreq-web.service   (slim FastAPI)         ├─ /highfreq*  → Tokyo:8000 via WG
─ neucast-paper-trader@btcusdt   (paper trader)         ├─ /grafana*   → Tokyo:3000 via WG
─ neucast-paper-trader@ethusdt                           │              + nginx Basic Auth + rate-limit
─ neucast-paper-trader@bnbusdt                           └─ /, /charts, /predict, … → Finland uvicorn
─ neucast-highfreq-trainer@*.timer (3 timers, 04:00)
─ neucast-l2-archive.timer        (02:00 UTC, → S3)   ─ neucast.service       (uvicorn, main webapp)
─ neucast-prom-backup.timer       (03:30 UTC, → S3)   ─ neucast-celery.service
─ prometheus + grafana + node-exporter                ─ Postgres 5433  (no HFT tables — dropped per ADR-009)
─ Postgres 5433 (single SoT for HFT data)             ─ wg0 [10.99.0.2]
─ wg0 [10.99.0.1]
─ /etc/neucast/env (chmod 600 root-only)
   ├─ DATABASE_URL + POSTGRES_PASSWORD
   ├─ HIGHFREQ_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT
   ├─ HIGHFREQ_STORE_L2_SNAPSHOTS=1
   ├─ YANDEX_S3_* (bucket + access keys)
   ├─ GRAFANA_ADMIN_PASSWORD
   └─ TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
```

See [ADR-009](../architecture.md#adr-009--tokyo-vps-as-the-hft-data-plane-supersedes-adr-006-for-the-hft-slice) (Tokyo placement),
[ADR-010](../architecture.md#adr-010--wireguard-tunnel-for-finland↔tokyo-http-traffic) (encryption),
[ADR-011](../architecture.md#adr-011--paper-trading-contract-time-stop-maker-only-fees-sim-only-by-construction) (paper trading).

## Deploy artefacts in this directory

| File | Role |
|---|---|
| **Bootstrap & app services** | |
| `bootstrap_tokyo.sh` | One-script bring-up of clean Ubuntu 22.04/24.04 → working ingest |
| `requirements-highfreq.txt` | Slim Python deps (no TF/Torch) — installed by bootstrap |
| `neucast-highfreq.service` | L2 ingest (always-on) — single-symbol legacy |
| `neucast-highfreq-web.service` | Slim FastAPI on 10.99.0.1:8000 (WG-only) |
| `neucast-paper-trader@.service` | Templated paper trader, one instance per symbol |
| `neucast-highfreq-trainer.service` + `.timer` | Single-symbol trainer (legacy) |
| `neucast-highfreq-trainer@.service` + `.timer` | Templated trainer for multi-symbol deploys |
| **Storage / archival** | |
| `neucast-l2-archive.service` + `.timer` | 02:00 UTC daily — L2 snapshots > 7 days → Yandex S3 |
| `neucast-prom-backup.service` + `.timer` | 03:30 UTC daily — Prometheus TSDB → Yandex S3 |
| **Monitoring (Prometheus + Grafana, hybrid)** | |
| `grafana/dashboards/hf-overview.json` | Auto-provisioned dashboard: 4 row groups (ingest/predictor/paper/system) |
| `grafana/alerting/alerts.yaml` | 5 alert rules with `{job="..."}` filters + custom thresholds |
| `grafana/alerting/contact-points.yaml` | Telegram contact point + HTML message template |
| **Networking** | |
| `wireguard_setup.md` | 10-step runbook for WG tunnel between Tokyo + Finland |

The Telegram contact point reads `HF_TELEGRAM_SIGNAL_BOT_TOKEN` and
`HF_TELEGRAM_SIGNAL_CHAT_ID` from the Grafana process environment. Keep the
real values in a root-only environment file on the deployment host; do not
write them into the provisioning YAML or commit them to Git.

## Production VPS layout (Hostkey Finland)

The production VPS predates Phase A and uses a hybrid layout:

```
/opt/neucast/                  # code dir, rsync'd from local worktree
├── app/                       # Python package
├── venv/                      # Python 3.10 venv (pip-installed)
├── docker-compose.yml         # only postgres+redis are managed via compose
└── docs/highfreq/             # architecture + deploy docs

/etc/systemd/system/
├── neucast.service                     # uvicorn (FastAPI) — pre-existing
├── neucast-celery.service              # celery worker — pre-existing
├── neucast-highfreq.service            # Phase A — L2 consumer (always-on)
├── neucast-highfreq-trainer.service    # Phase A — nightly trainer (oneshot)
└── neucast-highfreq-trainer.timer      # Phase A — fires .service at 04:00 UTC

# Bare docker-run containers (managed outside compose):
neucast-postgres   postgres:15-alpine   127.0.0.1:5433 → 5432
neucast-redis      redis:7-alpine       127.0.0.1:6380 → 6379
```

## First-time install

```bash
# 1. Sync code to /opt/neucast
rsync -av app/highfreq/ vps:/opt/neucast/app/highfreq/
rsync -av docs/highfreq/ vps:/opt/neucast/docs/highfreq/

# 2. Install asyncpg into the existing venv (no other new deps needed)
ssh vps '/opt/neucast/venv/bin/pip install "asyncpg>=0.29.0"'

# 3. Apply the SQL migration once
ssh vps 'sudo docker exec -i neucast-postgres psql -U neucast -d neucast' \
    < app/highfreq/migrations/001_initial_schema.sql

# 4. Install systemd unit (substitute ${POSTGRES_PASSWORD} first)
scp docs/highfreq/deploy/neucast-highfreq.service vps:/tmp/
ssh vps 'sudo install -m 0644 -o root -g root \
    /tmp/neucast-highfreq.service /etc/systemd/system/neucast-highfreq.service'

# 5. Enable + start the ingest unit
ssh vps 'sudo systemctl daemon-reload \
    && sudo systemctl enable --now neucast-highfreq.service'

# 6. Install the trainer unit + timer (substitute ${POSTGRES_PASSWORD})
scp docs/highfreq/deploy/neucast-highfreq-trainer.service vps:/tmp/
scp docs/highfreq/deploy/neucast-highfreq-trainer.timer   vps:/tmp/
ssh vps 'sudo install -m 0644 -o root -g root \
    /tmp/neucast-highfreq-trainer.service /etc/systemd/system/ \
    && sudo install -m 0644 -o root -g root \
    /tmp/neucast-highfreq-trainer.timer /etc/systemd/system/ \
    && sudo systemctl daemon-reload \
    && sudo systemctl enable --now neucast-highfreq-trainer.timer'

# Verify the timer is scheduled
ssh vps 'systemctl list-timers neucast-highfreq-trainer.timer --no-pager'
```

## Trainer cadence + exit codes

The `.service` is a `Type=oneshot` and the `.timer` fires it at 04:00 UTC
daily (with up to 5 min random jitter). The trainer's exit-code contract:

| Code | Meaning | systemd treats as |
|---|---|---|
| `0` | At least one walk-forward fold completed | success |
| `1` | "No folds yet" — not enough post-neutral-band bars (ramp-up) | success (via `SuccessExitStatus=0 1`) |
| `2` | Real error: DB unreachable, OOM, code bug | failed |

To run the trainer manually outside the timer:

```bash
ssh vps 'sudo systemctl start neucast-highfreq-trainer.service \
    && sudo journalctl -u neucast-highfreq-trainer.service -f'
```

To inspect the latest report:

```bash
ssh vps 'cat /opt/neucast/weights/highfreq/btcusdt_1m_metrics.json | jq .'
```

## Health check

```bash
# Service status + last 50 log lines
ssh vps 'sudo systemctl status neucast-highfreq.service --no-pager'
ssh vps 'sudo journalctl -u neucast-highfreq.service --since "5 minutes ago"'

# Throughput sanity (expect ~1 row/sec, lag <= flush_batch seconds)
ssh vps "sudo docker exec -i neucast-postgres psql -U neucast -d neucast -c \"
  SELECT count(*), max(ts) AS latest, EXTRACT(EPOCH FROM (now() - max(ts)))::int AS lag_s
  FROM highfreq_ofi_1s WHERE symbol='BTCUSDT' AND ts > now() - interval '5 minutes';\""
```

## Expected health log (every 30 s)

```
health: frames=346 snaps=291 trades=55 rows_emitted=32 rows_written=30 reconnects=0
```

For BTCUSDT @depth20@100ms:

* `snaps` ≈ 10/sec (one per 100 ms book frame)
* `trades` varies with volatility (typically 1-50/sec)
* `rows_emitted` ≈ 1/sec (1-second aggregation)
* `rows_written` ≈ rows_emitted (lags by `flush_batch_size` rows)
* `reconnects` should stay at 0 over 24 h+; >5/day means the WS is unstable
* `l2snaps_written` (when `HIGHFREQ_STORE_L2_SNAPSHOTS=1`) ≈ 1/sec/symbol

## Monitoring access (Prometheus + Grafana)

* **Grafana UI**: <https://neucast.ru/grafana> (basic-auth + admin login, 2 layers)
  * Dashboard: `NeuCast HF · Overview` (auto-provisioned, in `NeuCast` folder)
  * 4 row groups: ingest pipeline / predictor / paper trader / Tokyo system
* **Prometheus** (internal): `http://10.99.0.1:9099` — only reachable via WG
* **/metrics endpoints** (internal):
  * `:9090` ingest, `:9091/2/3` paper-traders (BTC/ETH/BNB), `:8000/metrics/` web, `:9100` node-exporter
* **Telegram alerts**: 5 rules wired to `telegram-stailfx` contact point with HTML template

```bash
# List active alerts via API
ADMIN_PASS=$(grep ^GRAFANA_ADMIN_PASSWORD /etc/neucast/env | cut -d= -f2-)
curl -s -u admin:$ADMIN_PASS http://10.99.0.1:3000/grafana/api/v1/provisioning/alert-rules | jq '.[].title'
```

## Yandex S3 archival check

```bash
ssh tokyo '
  source /etc/neucast/env
  /opt/neucast/venv/bin/python -c "
import os, boto3
s3 = boto3.client(\"s3\",
    endpoint_url=os.environ[\"YANDEX_S3_ENDPOINT\"],
    region_name=os.environ[\"YANDEX_S3_REGION\"],
    aws_access_key_id=os.environ[\"YANDEX_S3_ACCESS_KEY_ID\"],
    aws_secret_access_key=os.environ[\"YANDEX_S3_SECRET_ACCESS_KEY\"],
)
for prefix in (\"highfreq_l2/\", \"prometheus_snapshots/\"):
    r = s3.list_objects_v2(Bucket=os.environ[\"YANDEX_S3_BUCKET\"], Prefix=prefix)
    n = len(r.get(\"Contents\", []))
    size = sum(o[\"Size\"] for o in r.get(\"Contents\", [])) / 1024 / 1024
    print(f\"{prefix} : {n} objects, {size:.1f} MB\")
"'
```
