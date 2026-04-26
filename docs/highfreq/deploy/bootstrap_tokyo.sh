#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# NeuCast HFT — bootstrap a clean Tokyo VPS (4VPS.su JP-cx21 / Ubuntu 22.04 LTS)
#
# What this does:
#   1. Hardens the box (UFW, fail2ban, NTP — NTP is critical for HFT timestamps)
#   2. Creates the `stailfx` service user (matches existing systemd units)
#   3. Installs Docker (used only for Postgres, like prod Hostkey VPS)
#   4. Runs Postgres 15-alpine as a bare-docker container on 127.0.0.1:5433
#   5. Generates a strong POSTGRES_PASSWORD and stores it in /etc/neucast/env
#   6. Creates the `neucast` DB + role and applies the highfreq schema
#   7. Builds a minimal Python 3.10 venv at /opt/neucast/venv
#   8. Installs the systemd ingest unit + EnvironmentFile override
#   9. Enables + starts neucast-highfreq.service
#  10. Prints a health-check command to verify rows are landing
#
# Prerequisites (do these BEFORE running the script):
#   a. SSH into the VPS as root (4VPS gives root creds in their welcome email)
#   b. From your laptop, rsync the code to /opt/neucast:
#         rsync -av --exclude='.git' --exclude='__pycache__' \
#               --exclude='venv' --exclude='weights' \
#               ./ root@TOKYO_VPS:/opt/neucast/
#   c. Run this script as root:
#         bash /opt/neucast/docs/highfreq/deploy/bootstrap_tokyo.sh
#
# Idempotent: safe to re-run if something fails partway through.
# Logs to stdout; nothing is silently swallowed.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ─── Tunables ──────────────────────────────────────────────────────────────
NEUCAST_USER="${NEUCAST_USER:-stailfx}"
NEUCAST_DIR="${NEUCAST_DIR:-/opt/neucast}"
ENV_FILE="${ENV_FILE:-/etc/neucast/env}"
PG_CONTAINER="${PG_CONTAINER:-neucast-postgres}"
PG_HOST_PORT="${PG_HOST_PORT:-5433}"      # match prod Hostkey layout
PG_DB="${PG_DB:-neucast}"
PG_USER="${PG_USER:-neucast}"
PG_IMAGE="${PG_IMAGE:-postgres:15-alpine}"
# Use whatever python3 ships with the OS — 22.04 = 3.10, 24.04 = 3.12.
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
SSH_PORT="${SSH_PORT:-22}"

# ─── Helpers ───────────────────────────────────────────────────────────────
log()  { printf '\n\033[1;36m▸ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m! %s\033[0m\n'  "$*"; }
die()  { printf '\033[1;31m✗ %s\033[0m\n'  "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "Must run as root (use: sudo bash $0)"
[[ -f "$NEUCAST_DIR/app/highfreq/migrations/001_initial_schema.sql" ]] \
    || die "Code not found at $NEUCAST_DIR — rsync it from your laptop first (see header)"

# ─── 1. System update + base packages ──────────────────────────────────────
log "Updating apt + installing base packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
    ca-certificates curl gnupg lsb-release \
    ufw fail2ban \
    chrony \
    python3 python3-venv python3-pip python3-dev \
    build-essential pkg-config libssl-dev \
    git rsync vim less htop tmux jq traceroute net-tools

# ─── 2. NTP — critical for HFT timestamps ──────────────────────────────────
# We tag every L2 row with ts=exchange_event_time, but the local-recv-ms
# column uses the host clock. Drift > 1s would corrupt that diagnostic and
# make troubleshooting WS lag impossible.
log "Configuring chrony for NTP (Asia pool, fast sync)"
cat >/etc/chrony/conf.d/neucast-tokyo.conf <<'EOF'
# Asia-Pacific pool: closer than the default debian.pool, lower jitter.
pool 0.asia.pool.ntp.org iburst
pool 1.asia.pool.ntp.org iburst
pool 2.asia.pool.ntp.org iburst
# Step the clock if drift > 1s (only at startup); slew otherwise.
makestep 1.0 3
EOF
systemctl enable --now chrony
chronyc -a makestep >/dev/null || warn "chronyc makestep failed — non-fatal, will sync on next poll"
timedatectl set-timezone UTC
log "Time sync status:"
timedatectl | sed 's/^/    /'

# ─── 3. Firewall ───────────────────────────────────────────────────────────
log "Configuring UFW (allow only SSH inbound; postgres bound to 127.0.0.1 only)"
ufw --force reset >/dev/null
ufw default deny incoming
ufw default allow outgoing
ufw allow "$SSH_PORT/tcp" comment "SSH"
ufw --force enable
ufw status verbose | sed 's/^/    /'

# ─── 4. fail2ban (basic SSH bruteforce protection) ─────────────────────────
log "Enabling fail2ban for SSH"
systemctl enable --now fail2ban

# ─── 5. Service user ───────────────────────────────────────────────────────
if ! id -u "$NEUCAST_USER" >/dev/null 2>&1; then
    log "Creating service user: $NEUCAST_USER"
    useradd --system --shell /bin/bash --home-dir "$NEUCAST_DIR" --create-home "$NEUCAST_USER"
fi
chown -R "$NEUCAST_USER:$NEUCAST_USER" "$NEUCAST_DIR"

# ─── 6. Docker (used only for Postgres) ────────────────────────────────────
if ! command -v docker >/dev/null 2>&1; then
    log "Installing Docker CE (official repo)"
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
         https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
        > /etc/apt/sources.list.d/docker.list
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
    systemctl enable --now docker
else
    log "Docker already installed: $(docker --version)"
fi

# ─── 7. Postgres password (generate once, persist) ─────────────────────────
mkdir -p "$(dirname "$ENV_FILE")"
if [[ ! -f "$ENV_FILE" ]]; then
    log "Generating POSTGRES_PASSWORD (will be persisted at $ENV_FILE)"
    PG_PASS="$(openssl rand -base64 24 | tr -d '/+=' | cut -c1-28)"
    cat >"$ENV_FILE" <<EOF
# NeuCast HFT environment — used by systemd EnvironmentFile=
# Generated $(date -u +%FT%TZ) by bootstrap_tokyo.sh
PATH=${NEUCAST_DIR}/venv/bin:/usr/local/bin:/usr/bin:/bin
DATABASE_URL=postgresql://${PG_USER}:${PG_PASS}@127.0.0.1:${PG_HOST_PORT}/${PG_DB}
POSTGRES_PASSWORD=${PG_PASS}
HIGHFREQ_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT
HIGHFREQ_DEPTH_LEVELS=20
HIGHFREQ_UPDATE_SPEED_MS=100
HIGHFREQ_FLUSH_BATCH=1
LOG_LEVEL=INFO
TF_CPP_MIN_LOG_LEVEL=3
PYTHONDONTWRITEBYTECODE=1
EOF
    chmod 600 "$ENV_FILE"
    chown root:root "$ENV_FILE"
else
    log "Reusing existing $ENV_FILE (delete it to regenerate)"
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

# ─── 8. Postgres container ─────────────────────────────────────────────────
if ! docker ps -a --format '{{.Names}}' | grep -q "^${PG_CONTAINER}$"; then
    log "Starting Postgres container ($PG_IMAGE) on 127.0.0.1:$PG_HOST_PORT"
    docker run -d \
        --name "$PG_CONTAINER" \
        --restart unless-stopped \
        -p "127.0.0.1:${PG_HOST_PORT}:5432" \
        -e "POSTGRES_DB=${PG_DB}" \
        -e "POSTGRES_USER=${PG_USER}" \
        -e "POSTGRES_PASSWORD=${POSTGRES_PASSWORD}" \
        -v neucast-pgdata:/var/lib/postgresql/data \
        "$PG_IMAGE" \
        -c shared_buffers=256MB \
        -c work_mem=16MB \
        -c maintenance_work_mem=128MB \
        -c effective_cache_size=1GB \
        -c max_connections=50
else
    log "Postgres container already exists — ensuring it's running"
    docker start "$PG_CONTAINER" >/dev/null 2>&1 || true
fi

# Wait for postgres to be ready (max ~30 s)
log "Waiting for Postgres to accept connections..."
for i in $(seq 1 30); do
    if docker exec "$PG_CONTAINER" pg_isready -U "$PG_USER" -d "$PG_DB" >/dev/null 2>&1; then
        log "Postgres ready (took ~${i}s)"
        break
    fi
    sleep 1
    [[ $i -eq 30 ]] && die "Postgres did not become ready in 30s; check 'docker logs $PG_CONTAINER'"
done

# ─── 9. Apply HF schema ────────────────────────────────────────────────────
log "Applying highfreq migrations"
for migration in "$NEUCAST_DIR"/app/highfreq/migrations/*.sql; do
    [[ -f "$migration" ]] || continue
    log "  → $(basename "$migration")"
    docker exec -i "$PG_CONTAINER" psql -U "$PG_USER" -d "$PG_DB" -v ON_ERROR_STOP=1 \
        < "$migration" \
        | sed 's/^/      /'
done

# ─── 10. Python venv + minimal HFT deps ────────────────────────────────────
if [[ ! -d "$NEUCAST_DIR/venv" ]]; then
    log "Creating Python venv at $NEUCAST_DIR/venv"
    sudo -u "$NEUCAST_USER" "$PYTHON_BIN" -m venv "$NEUCAST_DIR/venv"
fi

log "Upgrading pip + installing HFT requirements (minimal — no TF/Torch)"
sudo -u "$NEUCAST_USER" "$NEUCAST_DIR/venv/bin/pip" install --quiet --upgrade pip
sudo -u "$NEUCAST_USER" "$NEUCAST_DIR/venv/bin/pip" install --quiet \
    -r "$NEUCAST_DIR/docs/highfreq/deploy/requirements-highfreq.txt"

# ─── 11. Systemd unit ──────────────────────────────────────────────────────
log "Installing neucast-highfreq.service"
# Strip the inline Environment= lines (they reference ${POSTGRES_PASSWORD}
# placeholders) and use EnvironmentFile= instead — cleaner than sed-substitution.
SRC_UNIT="$NEUCAST_DIR/docs/highfreq/deploy/neucast-highfreq.service"
DST_UNIT="/etc/systemd/system/neucast-highfreq.service"

# Generate a slimmed unit that points at our EnvironmentFile.
awk -v env_file="$ENV_FILE" '
    /^Environment=/ { next }   # drop inline env lines
    /^\[Service\]/ {
        print
        print "EnvironmentFile=" env_file
        next
    }
    { print }
' "$SRC_UNIT" > "$DST_UNIT"

chmod 0644 "$DST_UNIT"
chown root:root "$DST_UNIT"

systemctl daemon-reload
systemctl enable --now neucast-highfreq.service

# ─── 12. Verify ────────────────────────────────────────────────────────────
log "Service status (give it ~5s to connect to Binance):"
sleep 5
systemctl status neucast-highfreq.service --no-pager --lines=15 | sed 's/^/    /'

cat <<EOF

╔══════════════════════════════════════════════════════════════════════════╗
║  Bootstrap complete.                                                      ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  Next steps:                                                               ║
║                                                                            ║
║  1. Tail the live log to confirm WS frames arriving:                      ║
║       journalctl -u neucast-highfreq.service -f                            ║
║     Expect (every 30s):                                                    ║
║       health: frames=… snaps=… trades=… rows_emitted=… rows_written=…     ║
║                                                                            ║
║  2. Verify rows landing in Postgres (run after ~2 minutes):               ║
║       docker exec -i $PG_CONTAINER psql -U $PG_USER -d $PG_DB -c \\      ║
║         "SELECT count(*), max(ts) FROM highfreq_ofi_1s                    ║
║          WHERE ts > now() - interval '5 minutes';"                         ║
║                                                                            ║
║  3. Measure WS latency to Binance (do this from BOTH Tokyo and Finland   ║
║     to validate the Tokyo placement actually pays off):                   ║
║       python3 - <<'PY'                                                     ║
║       import asyncio, time, websockets                                     ║
║       async def m():                                                        ║
║         async with websockets.connect(                                     ║
║           "wss://stream.binance.com:9443/ws/btcusdt@depth20@100ms"        ║
║         ) as ws:                                                           ║
║           for _ in range(50):                                              ║
║             t0 = time.time(); await ws.recv()                              ║
║             print(f"{(time.time()-t0)*1000:.1f} ms")                       ║
║       asyncio.run(m())                                                     ║
║       PY                                                                   ║
║                                                                            ║
║  4. (Later, when ready to add the predictor + paper trader):              ║
║       Install the trainer + paper-trader systemd units the same way:      ║
║         cp docs/highfreq/deploy/neucast-highfreq-trainer.service \\       ║
║            /etc/systemd/system/ && systemctl daemon-reload                ║
║         systemctl enable --now neucast-highfreq-trainer.timer             ║
║                                                                            ║
║  Database password is in $ENV_FILE (chmod 600, root-only).                 ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════╝
EOF
