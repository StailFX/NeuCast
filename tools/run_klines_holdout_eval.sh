#!/bin/bash
# Klines holdout evaluation — proper "what does pretrain alone score?"
# experiment (T.15.i).
#
# Earlier T.15 / T.15.f / T.15.g experiments fine-tuned on the 93h live
# OFI window after pretrain — that's what produced the wide-CI numbers
# the user (rightly) flagged as "ужасные". This script does the
# scientifically clean thing: train CatBoost on the FIRST 80% of each
# Klines parquet, evaluate on the LAST 20%, report tight CI on the
# large test partition.
#
# Per-interval test-sample sizes (after target+neutral-band drop):
#
#   1m × 3y → ~1.17M bars → 234K test → CI ±0.2pp
#   5m × 3y → ~234K bars → 47K test → CI ±0.5pp
#   15m × 3y → ~78K bars → 15K test → CI ±0.8pp
#   30m × 3y → ~39K bars → 7.8K test → CI ±1.1pp
#   1h × 3y → ~19.5K bars → 3.9K test → CI ±1.6pp
#   4h × 3y → ~6.5K bars → 1.3K test → CI ±2.7pp
#   1d × 3y → ~1090 bars → 218 test → CI ±6.6pp
#
# Each invocation writes to weights/highfreq/candidate_klines_pretrain/
# overwriting earlier no-CV checkpoints with the same name. The new
# JSON metrics carry the holdout dir_acc + bootstrap CI.

ROOT=/opt/neucast
DATA_DIR=$ROOT/data/historical
PRETRAIN_DIR=$ROOT/weights/highfreq/candidate_klines_pretrain
LOG=/tmp/pipeline_holdout_eval.log

exec >> "$LOG" 2>&1
cd $ROOT
export PYTHONPATH=$ROOT

echo "================================================================"
echo "[$(date)] holdout-eval pipeline started"

_bar_minutes() {
  case "$1" in
    1m)  echo 1 ;;
    5m)  echo 5 ;;
    15m) echo 15 ;;
    30m) echo 30 ;;
    1h)  echo 60 ;;
    4h)  echo 240 ;;
    1d)  echo 1440 ;;
    *)   echo 1 ;;
  esac
}

# Run holdout on every interval × symbol combo where parquet exists.
for interval in 1m 5m 15m 30m 1h 4h 1d; do
  bar_minutes=$(_bar_minutes $interval)
  for sym in BTCUSDT ETHUSDT BNBUSDT; do
    symlower=$(echo $sym | tr A-Z a-z)
    f=$DATA_DIR/${symlower}_${interval}_klines.parquet
    out=$PRETRAIN_DIR/${symlower}_${interval}_pretrained.cbm
    if [ ! -f "$f" ]; then
      echo "[$(date)] WARN no parquet $f, skip"
      continue
    fi
    echo ""
    echo "[$(date)] === holdout-eval $sym @ $interval (bm=$bar_minutes) ==="
    /opt/neucast/venv/bin/python -m app.highfreq.pretrain \
      --data $f --symbol $sym \
      --bar-minutes $bar_minutes \
      --out $out \
      --cv-mode holdout \
      --holdout-test-fraction 0.20 \
      --iterations 800 --depth 5 --learning-rate 0.05 \
      || echo "[$(date)] WARN holdout-eval $sym @ $interval exited non-zero"
  done
done

# Aggregate JSON reports into one summary.
SUMMARY=/tmp/holdout_summary.txt
{
  echo "## Klines holdout evaluation (T.15.i)"
  echo ""
  echo "Train on first 80% of 3-year parquet, eval on last 20%."
  echo "Bootstrap 95% CI from the test partition."
  echo ""
  echo "| symbol | interval | n_test | dir_acc | CI_low | CI_high | base_rate |"
  echo "|--------|----------|--------|---------|--------|---------|-----------|"
  for interval in 1m 5m 15m 30m 1h 4h 1d; do
    for sym in btcusdt ethusdt bnbusdt; do
      json=$PRETRAIN_DIR/${sym}_${interval}_pretrained.json
      if [ -f "$json" ]; then
        sym_u=$(echo $sym | tr a-z A-Z)
        n_test=$(jq -r '
          if .n_bars_after_neutral_drop and .n_folds > 0
          then (.n_bars_after_neutral_drop * 0.20 | floor) | tostring
          else "—" end' $json)
        dir_acc=$(jq -r '.dir_acc_mean // "—" | if type == "number" then . * 100 | round / 100 else . end | tostring' $json)
        ci_lo=$(jq -r '.dir_acc_ci_low // "—" | if type == "number" then . * 100 | round / 100 else . end | tostring' $json)
        ci_hi=$(jq -r '.dir_acc_ci_high // "—" | if type == "number" then . * 100 | round / 100 else . end | tostring' $json)
        base_rate=$(jq -r '.base_rate // "—" | if type == "number" then . * 100 | round / 100 else . end | tostring' $json)
        echo "| $sym_u | $interval | $n_test | $dir_acc | $ci_lo | $ci_hi | $base_rate |"
      fi
    done
  done
} > $SUMMARY

cat $SUMMARY

# Telegram.
{
  echo "🧠 <b>NeuCast HF Klines holdout evaluation (T.15.i)</b>"
  echo ""
  echo "Honest answer: train on first 80% of 3-year parquet, eval on last 20%."
  echo "Tight CI on the large test partition."
  echo ""
  echo "<pre>"
  cat $SUMMARY
  echo "</pre>"
  echo ""
  echo "Run timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > /tmp/tg_body_holdout.html

/opt/neucast/venv/bin/python -m tools.telegram_notify \
  --from-file /tmp/tg_body_holdout.html \
  || echo "[$(date)] telegram notify failed (non-fatal)"

echo "[$(date)] holdout-eval pipeline DONE"
