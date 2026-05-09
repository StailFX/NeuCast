#!/usr/bin/env bash
# Wrapper invoked by neucast-defence-figures.service.
#
# Pipeline (runs daily 05:00 UTC, ~10 min after the joint trainer):
#   1. Pull live conditional accuracy from the local Tokyo FastAPI
#      (no auth needed — it's on the WireGuard 10.99.0.1 interface).
#   2. Render 5 SVG figures from the latest training-run JSONs into
#      a stable local dir at /opt/neucast/weights/highfreq/figures/.
#      The operator/dev pulls them with one scp when slides need
#      a refresh — keeps Tokyo→Finland out of the trust path.
#   3. Emit a textfile_collector heartbeat for the Grafana
#      stale-cron alert framework.
#
# Exit codes:
#   0 = render succeeded
#   ≥1 = render itself failed
#
# Operator note: re-running this manually is safe — it overwrites
# the figure set in place. To refresh slides:
#   scp root@147.45.49.40:/opt/neucast/weights/highfreq/figures/*.svg \
#       docs/defence/figures/

set -euo pipefail

TS=$(date -u +%FT%TZ)
OUT=/opt/neucast/weights/highfreq/figures

echo "[run_defence_figures] $TS — rendering to $OUT"
mkdir -p "$OUT"

# ── 1. Pull live conditional accuracy from local FastAPI ──
# Slim HF API listens on 10.99.0.1:8000 (WireGuard endpoint). No
# basic auth on this side — that's nginx-side on Finland.
COND_JSON=/tmp/conditional_accuracy.json
if ! curl -s --max-time 10 \
    http://10.99.0.1:8000/api/highfreq/conditional_accuracy \
    > "$COND_JSON"; then
    echo "[run_defence_figures] WARN: conditional_accuracy fetch failed; using stale" >&2
fi

# ── 2. Render figures ──
# We're already running as stailfx (see neucast-defence-figures.service
# `User=stailfx`), so just exec directly — no sudo dance, which would
# trip systemd's NoNewPrivileges hardening anyway.
cd /opt/neucast
/opt/neucast/venv/bin/python -m tools.render_defence_figures \
    --json-dir /opt/neucast/weights/highfreq \
    --conditional-accuracy "$COND_JSON" \
    --out "$OUT"

ls -la "$OUT"

# ── 3. Heartbeat ──
HEARTBEAT_DIR=/var/lib/prometheus/node-exporter
HEARTBEAT=$HEARTBEAT_DIR/neucast_hf_defence_figures.prom
TMP_HB=$(mktemp "$HEARTBEAT.XXXXXX")
{
    echo "# HELP neucast_hf_defence_figures_last_success_timestamp_seconds Last successful render (unix seconds)"
    echo "# TYPE neucast_hf_defence_figures_last_success_timestamp_seconds gauge"
    echo "neucast_hf_defence_figures_last_success_timestamp_seconds $(date +%s)"
} > "$TMP_HB"
mv "$TMP_HB" "$HEARTBEAT"

echo "[run_defence_figures] done"
