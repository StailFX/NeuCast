#!/bin/bash
# All-remaining-horizons Klines pretrain pipeline (T.15.g).
#
# After T.15 (1m fail) and T.15.f (5m+15m mixed), this covers the
# remaining liquid trading intervals: 30m, 1h, 4h, 1d.
#
# Per-interval treatment depends on live-data feasibility:
#
#   30m → pretrain + fine-tune + compare  (186 bars from 93h live)
#   1h  → pretrain + fine-tune + compare  ( 93 bars — borderline)
#   4h  → pretrain ONLY (insufficient live for walk-forward CV)
#   1d  → pretrain ONLY (~4 bars from 93h live, useless for fit)
#
# The 4h/1d pretrained checkpoints are saved as academic reference —
# they're effectively "what does CatBoost think about <symbol> at
# this horizon given 3 years of Klines OHLC?" — usable for one-shot
# analysis even if not deployable to live paper-trader.
#
# Required env (passed via sudo --preserve-env):
#   DATABASE_URL, HF_TELEGRAM_SIGNAL_BOT_TOKEN, HF_TELEGRAM_SIGNAL_CHAT_ID,
#   HF_TELEGRAM_SIGNAL_ENABLED.
#
# Usage (root shell with /etc/neucast/env sourced):
#   set -a; source /etc/neucast/env; set +a
#   sudo --preserve-env=DATABASE_URL,HF_TELEGRAM_SIGNAL_BOT_TOKEN,HF_TELEGRAM_SIGNAL_CHAT_ID,HF_TELEGRAM_SIGNAL_ENABLED \
#        -u stailfx bash /opt/neucast/tools/run_all_horizons_pipeline.sh

ROOT=/opt/neucast
DATA_DIR=$ROOT/data/historical
PRETRAIN_DIR=$ROOT/weights/highfreq/candidate_klines_pretrain
FINETUNE_DIR=$ROOT/weights/highfreq/candidate_finetune_multihorizon
PROD_DIR=$ROOT/weights/highfreq
LOG=/tmp/pipeline_all_horizons.log

mkdir -p "$FINETUNE_DIR"
exec >> "$LOG" 2>&1

cd $ROOT
export PYTHONPATH=$ROOT

echo "================================================================"
echo "[$(date)] all-horizons pipeline started"

# ── Step 1: download Klines for all 4 new intervals ─────────────────
echo "[$(date)] downloading 30m, 1h, 4h, 1d klines (3 years × 3 symbols)…"
for interval in 30m 1h 4h 1d; do
  for sym in BTCUSDT ETHUSDT BNBUSDT; do
    symlower=$(echo $sym | tr A-Z a-z)
    f=$DATA_DIR/${symlower}_${interval}_klines.parquet
    if [ -f "$f" ]; then
      echo "[$(date)] skip $sym @ $interval (parquet exists)"
      continue
    fi
    echo "[$(date)] downloading $sym @ $interval"
    /opt/neucast/venv/bin/python -m tools.binance_klines_download \
      --symbol $sym \
      --years 3 \
      --interval $interval \
      --out-dir $DATA_DIR \
      || echo "[$(date)] WARN download $sym @ $interval failed"
  done
done
echo "[$(date)] downloads complete"

# ── Helper: compute --bar-minutes from interval string ──────────────
_bar_minutes() {
  case "$1" in
    30m) echo 30 ;;
    1h)  echo 60 ;;
    4h)  echo 240 ;;
    1d)  echo 1440 ;;
    *)   echo 1 ;;
  esac
}

# ── Step 2: pretrain all 4 intervals × 3 symbols ───────────────────
for interval in 30m 1h 4h 1d; do
  bar_minutes=$(_bar_minutes $interval)
  for sym in BTCUSDT ETHUSDT BNBUSDT; do
    symlower=$(echo $sym | tr A-Z a-z)
    f=$DATA_DIR/${symlower}_${interval}_klines.parquet
    out=$PRETRAIN_DIR/${symlower}_${interval}_pretrained.cbm
    if [ ! -f "$f" ]; then
      echo "[$(date)] WARN no parquet $f, skip pretrain"
      continue
    fi
    echo "[$(date)] --- pretrain $sym @ $interval (bm=$bar_minutes) ---"
    /opt/neucast/venv/bin/python -m app.highfreq.pretrain \
      --data $f --symbol $sym \
      --bar-minutes $bar_minutes \
      --out $out \
      --no-walk-forward \
      --iterations 800 --depth 5 --learning-rate 0.05 \
      || echo "[$(date)] WARN pretrain $sym @ $interval exited non-zero"
  done
done

# ── Step 3: fine-tune only on intervals with enough live data ──────
# Live OFI seconds depth ≈ 93h → bar count drops with bar size. Scale
# the walk-forward params by bar size so we get a usable fold count
# despite the small total-bar count.
#
# 30m: 186 bars total — initial=90 bars, test=1 bar, step=1 bar, min=80 → ~96 folds
# 1h:   93 bars total — initial=45 bars, test=1 bar, step=1 bar, min=40 → ~48 folds
# 4h, 1d skipped (not enough bars even with min_train=20).
#
# Tiny per-fold sample means individual fold dir_acc is noisy — but
# pooling 50-100 folds gives reasonable CI for an honest estimate.
for interval in 30m 1h; do
  bar_minutes=$(_bar_minutes $interval)
  case $interval in
    30m) initial_train=2700; test_fold=30; step=30; min_train=80 ;;   # 90 bars init
    1h)  initial_train=2700; test_fold=60; step=60; min_train=40 ;;   # 45 bars init
  esac
  for sym in BTCUSDT ETHUSDT BNBUSDT; do
    symlower=$(echo $sym | tr A-Z a-z)
    pretrain_path=$PRETRAIN_DIR/${symlower}_${interval}_pretrained.cbm
    cand_path=$FINETUNE_DIR/${symlower}_${interval}.cbm
    metrics_path=$FINETUNE_DIR/${symlower}_${interval}_metrics.json
    if [ ! -f "$pretrain_path" ]; then
      echo "[$(date)] WARN no pretrain ckpt for $sym @ $interval, skip fine-tune"
      continue
    fi
    echo "[$(date)] --- fine-tune $sym @ $interval (initial=$initial_train min=$min_train) ---"
    /opt/neucast/venv/bin/python -m app.highfreq.trainer \
      --symbol $sym \
      --since-hours 93 \
      --bar-minutes $bar_minutes \
      --feature-set long_horizon \
      --init-from $pretrain_path \
      --out $cand_path \
      --report $metrics_path \
      --frozen-holdout-days 0 \
      --sample-weight-half-life-bars 0 \
      --initial-train-minutes $initial_train \
      --test-fold-minutes $test_fold \
      --step-minutes $step \
      --min-train-samples $min_train \
      || echo "[$(date)] WARN fine-tune $sym @ $interval exited non-zero"
  done
done

# ── Step 4: comparator per interval (using filename shim) ──────────
COMPARE_OUT=/tmp/compare_all_horizons.txt
> $COMPARE_OUT

for interval in 30m 1h; do
  shim_prod=$(mktemp -d)
  shim_cand=$(mktemp -d)
  for sym in btcusdt ethusdt bnbusdt; do
    if [ -f "$PROD_DIR/${sym}_${interval}_metrics.json" ]; then
      cp $PROD_DIR/${sym}_${interval}_metrics.json $shim_prod/${sym}_1m_metrics.json
    fi
    if [ -f "$FINETUNE_DIR/${sym}_${interval}_metrics.json" ]; then
      cp $FINETUNE_DIR/${sym}_${interval}_metrics.json $shim_cand/${sym}_1m_metrics.json
    fi
  done
  echo "" >> $COMPARE_OUT
  echo "### Interval $interval (fine-tune)" >> $COMPARE_OUT
  /opt/neucast/venv/bin/python -m tools.compare_candidate_models \
    --production-dir $shim_prod \
    --candidate-dir $shim_cand \
    --tolerance 0.005 \
    >> $COMPARE_OUT 2>&1 || true
  rm -rf $shim_prod $shim_cand
done

# Pretrain-only summary for 4h / 1d (no fine-tune, no comparator):
{
  echo ""
  echo "### Pretrained-only checkpoints (4h, 1d) — for academic reference"
  echo ""
  for interval in 4h 1d; do
    for sym in btcusdt ethusdt bnbusdt; do
      f=$PRETRAIN_DIR/${sym}_${interval}_pretrained.cbm
      json=$PRETRAIN_DIR/${sym}_${interval}_pretrained.json
      if [ -f "$json" ]; then
        n_bars=$(jq -r '.n_bars_after_neutral_drop // "?"' "$json")
        bm=$(jq -r '.bar_minutes // "?"' "$json")
        echo "$sym @ $interval: $n_bars bars after neutral-band drop (bar_minutes=$bm)"
      fi
    done
  done
} >> $COMPARE_OUT

# ── Step 5: Telegram notification ──────────────────────────────────
{
  echo "🧠 <b>NeuCast HF ALL-HORIZONS pretrain → fine-tune complete (T.15.g)</b>"
  echo ""
  echo "Trained 30m / 1h / 4h / 1d on 3 years of Klines."
  echo ""
  echo "<pre>"
  cat $COMPARE_OUT
  echo "</pre>"
  echo ""
  echo "<b>Notes:</b>"
  echo "• 4h and 1d are pretrain-only (insufficient live data for fine-tune walk-forward CV)."
  echo "• Pretrain checkpoints saved in <code>$PRETRAIN_DIR</code>"
  echo "• Fine-tuned candidates in <code>$FINETUNE_DIR</code>"
  echo ""
  echo "Run timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > /tmp/tg_body_all_horizons.html

/opt/neucast/venv/bin/python -m tools.telegram_notify \
  --from-file /tmp/tg_body_all_horizons.html \
  || echo "[$(date)] telegram notify failed (non-fatal)"

echo "[$(date)] all-horizons pipeline DONE"
