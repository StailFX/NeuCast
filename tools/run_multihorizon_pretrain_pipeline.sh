#!/bin/bash
# Multi-horizon Klines pretrain → fine-tune pipeline (T.15.f).
#
# Follow-up to the failed 1m experiment. Hypothesis: at 5m / 15m bars
# the long_horizon TA pipeline becomes competitive (microstructure
# OFI signal decays at this scale), so a 3-year pretrain + fine-tune
# may actually beat the existing 15m production model.
#
# Sequence per interval (5m, 15m):
#   1. Wait for that interval's parquets to exist for all 3 symbols.
#   2. Pretrain CatBoost on Klines parquet (long_horizon features).
#   3. Fine-tune via trainer --init-from --bar-minutes <N>.
#   4. Comparator vs production weights at the same interval.
# Then send Telegram with unified verdict per (symbol × interval).
#
# Required env (passed via sudo --preserve-env from caller):
#   DATABASE_URL, HF_TELEGRAM_SIGNAL_*
#
# Usage (root shell with /etc/neucast/env sourced):
#   set -a; source /etc/neucast/env; set +a
#   sudo --preserve-env=DATABASE_URL,HF_TELEGRAM_SIGNAL_BOT_TOKEN,HF_TELEGRAM_SIGNAL_CHAT_ID,HF_TELEGRAM_SIGNAL_ENABLED \
#        -u stailfx bash /opt/neucast/tools/run_multihorizon_pretrain_pipeline.sh
#
# Idempotent: re-running overwrites existing candidate files.

ROOT=/opt/neucast
DATA_DIR=$ROOT/data/historical
PRETRAIN_DIR=$ROOT/weights/highfreq/candidate_klines_pretrain
FINETUNE_DIR=$ROOT/weights/highfreq/candidate_finetune_multihorizon
PROD_DIR=$ROOT/weights/highfreq
LOG=/tmp/pipeline_multihorizon.log

mkdir -p "$FINETUNE_DIR"
exec >> "$LOG" 2>&1

cd $ROOT
export PYTHONPATH=$ROOT

echo "================================================================"
echo "[$(date)] multihorizon pipeline started"

# ── Step 1: wait for klines downloader to finish ────────────────────
echo "[$(date)] waiting for klines downloader to exit…"
while pgrep -f binance_klines_download >/dev/null; do
  sleep 30
done
echo "[$(date)] klines downloader exited"

# ── Step 2: per-interval pretrain + fine-tune ──────────────────────
COMPARE_OUT=/tmp/compare_multihorizon.txt
> $COMPARE_OUT

for interval in 5m 15m; do
  bar_minutes=${interval%m}
  echo ""
  echo "================================================================"
  echo "[$(date)] === interval=$interval (bar_minutes=$bar_minutes) ==="

  # Verify all 3 parquets exist for this interval
  missing=0
  for sym in btcusdt ethusdt bnbusdt; do
    f=$DATA_DIR/${sym}_${interval}_klines.parquet
    if [ ! -f "$f" ]; then
      echo "[$(date)] ERROR missing parquet $f"
      missing=1
    fi
  done
  if [ $missing -eq 1 ]; then
    echo "[$(date)] skipping interval $interval — parquets missing"
    continue
  fi

  # Pretrain each symbol at this interval.
  for sym in BTCUSDT ETHUSDT BNBUSDT; do
    symlower=$(echo $sym | tr A-Z a-z)
    pretrain_path=$PRETRAIN_DIR/${symlower}_${interval}_pretrained.cbm
    echo "[$(date)] --- pretrain $sym @ $interval ---"
    /opt/neucast/venv/bin/python -m app.highfreq.pretrain \
      --data $DATA_DIR/${symlower}_${interval}_klines.parquet \
      --symbol $sym \
      --bar-minutes $bar_minutes \
      --out $pretrain_path \
      --no-walk-forward \
      --iterations 800 --depth 5 --learning-rate 0.05 \
      || echo "[$(date)] WARN pretrain $sym @ $interval exited non-zero"
  done

  # Fine-tune each symbol at this interval, writing to candidate dir.
  # Output filename embeds bar_minutes so we don't overwrite 1m candidates.
  for sym in BTCUSDT ETHUSDT BNBUSDT; do
    symlower=$(echo $sym | tr A-Z a-z)
    pretrain_path=$PRETRAIN_DIR/${symlower}_${interval}_pretrained.cbm
    candidate_path=$FINETUNE_DIR/${symlower}_${interval}.cbm
    metrics_path=$FINETUNE_DIR/${symlower}_${interval}_metrics.json
    if [ ! -f "$pretrain_path" ]; then
      echo "[$(date)] WARN no pretrain ckpt for $sym @ $interval, skipping fine-tune"
      continue
    fi
    echo "[$(date)] --- fine-tune $sym @ $interval ---"
    /opt/neucast/venv/bin/python -m app.highfreq.trainer \
      --symbol $sym \
      --since-hours 93 \
      --bar-minutes $bar_minutes \
      --feature-set long_horizon \
      --init-from $pretrain_path \
      --out $candidate_path \
      --report $metrics_path \
      --frozen-holdout-days 0 \
      --sample-weight-half-life-bars 0 \
      || echo "[$(date)] WARN fine-tune $sym @ $interval exited non-zero"
  done

  # Comparator at this interval. Note prod weights are named
  # <sym>_15m.cbm (no 5m prod exists), so for 5m we'll see 'missing'
  # for production rows — that's honest.
  echo "[$(date)] --- compare $interval ---"
  echo "" >> $COMPARE_OUT
  echo "## $interval" >> $COMPARE_OUT
  /opt/neucast/venv/bin/python -m tools.compare_candidate_models \
    --production-dir $PROD_DIR \
    --candidate-dir $FINETUNE_DIR \
    --tolerance 0.005 \
    --symbol BTCUSDT --symbol ETHUSDT --symbol BNBUSDT \
    >> $COMPARE_OUT 2>&1 || true
  # NOTE: comparator's compare_one looks for <sym>_1m_metrics.json;
  # need to call with interval-aware paths. For now we write a
  # compatibility shim: copy metrics with `_<interval>` suffix to
  # `_1m` for comparator to pick up. But that breaks for multi-call.
  # Simpler: re-run comparator below per-interval with a temp dir.
done

# ── Step 3: per-interval comparator using interval-specific metric dirs ─
# Build temp dirs that the comparator can consume (it expects
# <sym>_1m_metrics.json filename pattern).
echo ""
echo "[$(date)] === per-interval comparator (with file-naming shim) ==="
> $COMPARE_OUT
for interval in 5m 15m; do
  shim_prod=$(mktemp -d)
  shim_cand=$(mktemp -d)
  for sym in btcusdt ethusdt bnbusdt; do
    # Production weights for 5m don't exist; the comparator emits
    # 'missing' rows in that case (correct behaviour).
    if [ -f "$PROD_DIR/${sym}_${interval}_metrics.json" ]; then
      cp $PROD_DIR/${sym}_${interval}_metrics.json $shim_prod/${sym}_1m_metrics.json
    fi
    if [ -f "$FINETUNE_DIR/${sym}_${interval}_metrics.json" ]; then
      cp $FINETUNE_DIR/${sym}_${interval}_metrics.json $shim_cand/${sym}_1m_metrics.json
    fi
  done
  echo "" >> $COMPARE_OUT
  echo "### Interval $interval" >> $COMPARE_OUT
  /opt/neucast/venv/bin/python -m tools.compare_candidate_models \
    --production-dir $shim_prod \
    --candidate-dir $shim_cand \
    --tolerance 0.005 \
    >> $COMPARE_OUT 2>&1 || true
  rm -rf $shim_prod $shim_cand
done

# ── Step 4: Telegram notification ──────────────────────────────────
{
  echo "🧠 <b>NeuCast HF MULTIHORIZON pretrain → fine-tune complete (T.15.f)</b>"
  echo ""
  echo "Hypothesis: at 5m/15m bars long_horizon TA may beat production."
  echo ""
  echo "<pre>"
  cat $COMPARE_OUT
  echo "</pre>"
  echo ""
  echo "Pretrain checkpoints in <code>$PRETRAIN_DIR</code>"
  echo "Fine-tune candidates in <code>$FINETUNE_DIR</code>"
  echo "Run timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > /tmp/tg_body_multihorizon.html

/opt/neucast/venv/bin/python -m tools.telegram_notify \
  --from-file /tmp/tg_body_multihorizon.html \
  || echo "[$(date)] telegram notify failed (non-fatal)"

echo "[$(date)] multihorizon pipeline DONE"
