#!/usr/bin/env bash
# Wrapper invoked by neucast-highfreq-trainer-multihorizon@<inst>.service.
#
# Parses the systemd instance name `<symbol>-h<horizon>m` (e.g.
# "btcusdt-h15m", "ethusdt-h60m") and execs the trainer with the
# right --symbol / --bar-minutes / --since-hours / output-path args.
# Lives outside the unit file because systemd's ${} parameter expansion
# would clobber bash's ${} expansion if inlined.
#
# Exit codes:
#   0    — trainer succeeded with at least one fold (or trainer's "1"
#          early-bootstrap exit, which the unit file maps to ok)
#   64   — usage error (instance name didn't parse)
#   *    — trainer's own failure code

set -euo pipefail

inst="${1:?usage: run_trainer_multihorizon.sh <symbol>-h<horizon>m}"

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

# Validate horizon is a positive integer.
case "$horizon" in
    "" | *[!0-9]*)
        echo "horizon must be positive integer, got: $horizon" >&2
        exit 64
        ;;
esac

# Scale --since-hours by horizon. 1m needs 72h for n>2000;
# 15m at 72h gives only ~288 bars (4 days × 96), too thin;
# 60m at 240h gives ~240 bars, still thin → bump to 720h.
if [ "$horizon" -le 4 ]; then
    since_hours=72
elif [ "$horizon" -le 15 ]; then
    since_hours=240
else
    since_hours=720
fi

out="/opt/neucast/weights/highfreq/${sym}_${horizon}m.cbm"
report="/opt/neucast/weights/highfreq/${sym}_${horizon}m_metrics.json"

echo "[run_trainer_multihorizon] sym=$sym horizon=${horizon}m since_hours=$since_hours" >&2
echo "[run_trainer_multihorizon] out=$out" >&2

exec /opt/neucast/venv/bin/python -m app.highfreq.trainer \
    --symbol "$sym" \
    --bar-minutes "$horizon" \
    --since-hours "$since_hours" \
    --out "$out" \
    --report "$report" \
    --frozen-holdout-days 0
