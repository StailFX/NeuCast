#!/usr/bin/env bash
# Wrapper invoked by neucast-paper-trader-multihorizon@<inst>.service.
#
# Parses the systemd instance name `<symbol>-h<horizon>m` (e.g.
# "btcusdt-h15m") and exports HIGHFREQ_PAPER_SYMBOL +
# HF_HORIZON_MINUTES before exec'ing the runner. Lives outside the
# unit file because systemd's ${} parameter expansion would clobber
# bash's ${} expansion if inlined.
#
# Exit codes:
#   64  — usage error (instance name didn't parse)
#   *   — runner's own exit code (always-restart unless killed)

set -euo pipefail

inst="${1:?usage: run_paper_trader_multihorizon.sh <symbol>-h<horizon>m}"

# Validate format.
case "$inst" in
    *-h*m) ;;
    *)
        echo "instance must be <symbol>-h<horizon>m, got: $inst" >&2
        exit 64
        ;;
esac

sym="${inst%%-h*}"
rest="${inst#*-h}"
horizon="${rest%m}"

case "$horizon" in
    "" | *[!0-9]*)
        echo "horizon must be positive integer, got: $horizon" >&2
        exit 64
        ;;
esac

export HIGHFREQ_PAPER_SYMBOL="$sym"
export HF_HORIZON_MINUTES="$horizon"

# Prometheus metrics port. The three 1-minute runners hold 9091/9092/9093.
# Multi-horizon runners get 9094-9099. Map deterministically:
#   bnb=0, btc=1, eth=2, ...   (other symbols → fall through to 0)
#   port = 9094 + (sym_idx * 3 + horizon_class)
# horizon_class: 0 for ≤5m, 1 for 6-30m, 2 for >30m.
case "$sym" in
    bnb*) sym_idx=0 ;;
    btc*) sym_idx=1 ;;
    eth*) sym_idx=2 ;;
    *)    sym_idx=0 ;;
esac
if [ "$horizon" -le 5 ]; then
    horizon_class=0
elif [ "$horizon" -le 30 ]; then
    horizon_class=1
else
    horizon_class=2
fi
port=$((9094 + sym_idx * 3 + horizon_class))
export HIGHFREQ_PAPER_METRICS_PORT="$port"

# Tighter entry thresholds for long-horizon runners — the per-bar
# decision is rarer (every 15-60 min vs every minute) so we accept
# only stronger signals to avoid burning fee budget on the noise.
# Operator can override via env if needed (env wins, by env-var
# precedence — these are defaults the wrapper provides).
: "${HF_ENTRY_LONG_THRESHOLD:=0.60}"
: "${HF_ENTRY_SHORT_THRESHOLD:=0.40}"
export HF_ENTRY_LONG_THRESHOLD HF_ENTRY_SHORT_THRESHOLD

echo "[run_paper_trader_multihorizon] sym=$sym horizon=${horizon}m port=$port" \
    "long_threshold=$HF_ENTRY_LONG_THRESHOLD short_threshold=$HF_ENTRY_SHORT_THRESHOLD" >&2

exec /opt/neucast/venv/bin/python -m app.highfreq.paper_trader_runner
