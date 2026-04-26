# Phase A — Deployment notes

This directory contains deploy artifacts referenced by
[`../architecture.md`](../architecture.md). Three deployment paths exist; pick
based on where you're starting from.

| Mode | Where | When |
|---|---|---|
| **Tokyo greenfield (ADR-009)** | `bootstrap_tokyo.sh` | Fresh Ubuntu 22.04/24.04 VPS in Tokyo — single command from clean OS to running ingest + slim web. **This is the current production path.** |
| Docker Compose | `docker-compose.yml` → `highfreq-l2` service | Local dev only |
| Systemd (legacy) | `neucast-highfreq.service` (this dir) | Existing Hostkey Finland VPS where app/celery already run as systemd units. Per ADR-009, ingest no longer runs here — kept for reference. |

## Production layout (post ADR-009)

```
TOKYO 147.45.49.40 (4VPS.su JP-cx21, Ubuntu 24.04)    FINLAND 151.245.139.21 (Hostkey)
├─ neucast-highfreq.service       (L2 ingest)         ├─ nginx → reverse-proxies /highfreq* to Tokyo:8000
├─ neucast-highfreq-web.service   (slim FastAPI)      ├─ neucast.service       (uvicorn, main webapp)
├─ Postgres 5433  (single source of truth)            ├─ neucast-celery.service
├─ UFW: 22 = world, 8000 = Finland-only               └─ Postgres 5433  (no highfreq tables)
└─ /etc/neucast/env (DATABASE_URL + POSTGRES_PASSWORD)
```

See [ADR-009 in architecture.md](../architecture.md#adr-009--tokyo-vps-as-the-hft-data-plane-supersedes-adr-006-for-the-hft-slice) for the rationale.

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
