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

# Long-horizon trade economics differ from 1m: each bar's decision is
# rarer (every 15-60 min vs every minute) and fee burden per trade is
# the same. Conclusion: tighter conviction filter pays off — fewer
# trades but each carries higher expected magnitude.
#
# Per-horizon defaults (wrapper-driven, NOT env-driven, because the
# 1m production runners use 0.55/0.45 and we want long-horizon runners
# DEFAULT differently without touching /etc/neucast/env globally):
#   1-4m:    0.55 / 0.45  (production 1m contract — original)
#   5-30m:   0.65 / 0.35  (15m: dir_acc 0.66 → high-conviction subset
#                          has E[|move|] ≈ 7-10 bp, edge × move ~3 bp)
#   31m+:    0.70 / 0.30  (60m: small n, want strong signals only)
#
# Operator override path: set HF_FORCE_LONG_THRESHOLD / HF_FORCE_SHORT_THRESHOLD
# in the systemd unit drop-in. These are NEW variable names that
# /etc/neucast/env doesn't carry, so dropin Environment= wins cleanly.
if [ -n "${HF_FORCE_LONG_THRESHOLD:-}" ]; then
    HF_ENTRY_LONG_THRESHOLD="$HF_FORCE_LONG_THRESHOLD"
elif [ "$horizon" -le 4 ]; then
    HF_ENTRY_LONG_THRESHOLD=0.55
elif [ "$horizon" -le 30 ]; then
    HF_ENTRY_LONG_THRESHOLD=0.65
else
    HF_ENTRY_LONG_THRESHOLD=0.70
fi
if [ -n "${HF_FORCE_SHORT_THRESHOLD:-}" ]; then
    HF_ENTRY_SHORT_THRESHOLD="$HF_FORCE_SHORT_THRESHOLD"
elif [ "$horizon" -le 4 ]; then
    HF_ENTRY_SHORT_THRESHOLD=0.45
elif [ "$horizon" -le 30 ]; then
    HF_ENTRY_SHORT_THRESHOLD=0.35
else
    HF_ENTRY_SHORT_THRESHOLD=0.30
fi
export HF_ENTRY_LONG_THRESHOLD HF_ENTRY_SHORT_THRESHOLD

# Disable demo mode for long-horizon runners — the model's calibration
# gate is meaningful here (long-horizon models are produced one per
# day max, vs 1m which trains hourly) and we want realized accuracy
# to flow into the official stats. /etc/neucast/env may have
# HF_PAPER_DEMO_MODE=1 set globally for legacy reasons; this wins
# (we export AFTER any sourcing).
export HF_PAPER_DEMO_MODE=0

echo "[run_paper_trader_multihorizon] sym=$sym horizon=${horizon}m port=$port" \
    "long_threshold=$HF_ENTRY_LONG_THRESHOLD short_threshold=$HF_ENTRY_SHORT_THRESHOLD" \
    "demo=$HF_PAPER_DEMO_MODE" >&2

exec /opt/neucast/venv/bin/python -m app.highfreq.paper_trader_runner
