# Phase A — Deployment notes

This directory contains deploy artifacts referenced by
[`../architecture.md`](../architecture.md) (ADR-006). Two deployment
modes are supported:

| Mode | Where | When |
|---|---|---|
| Docker Compose | `docker-compose.yml` → `highfreq-l2` service | Local dev; greenfield VPS |
| Systemd | `neucast-highfreq.service` (this dir) | Existing NeuCast VPS where app/celery already run as systemd units |

## Production VPS layout (Hostkey Finland)

The production VPS predates Phase A and uses a hybrid layout:

```
/opt/neucast/                  # code dir, rsync'd from local worktree
├── app/                       # Python package
├── venv/                      # Python 3.10 venv (pip-installed)
├── docker-compose.yml         # only postgres+redis are managed via compose
└── docs/highfreq/             # architecture + deploy docs

/etc/systemd/system/
├── neucast.service            # uvicorn (FastAPI) — pre-existing
├── neucast-celery.service     # celery worker — pre-existing
└── neucast-highfreq.service   # NEW: Phase A L2 consumer

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

# 5. Enable + start
ssh vps 'sudo systemctl daemon-reload \
    && sudo systemctl enable --now neucast-highfreq.service'
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
