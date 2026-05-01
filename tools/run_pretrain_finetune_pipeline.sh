#!/bin/bash
# Full HF candidate-model pipeline runner with Telegram notification.
#
# Sequence:
#   1. Wait for pretrain pipeline to exit.
#   2. Verify all 3 pretrained .cbm checkpoints exist.
#   3. Fine-tune via trainer --init-from for each symbol on recent live
#      OFI data, writing to candidate_finetune/ (not production!).
#   4. Run candidate comparator.
#   5. Send Telegram notification with summary.
#
# Required env (passed via sudo --preserve-env):
#   DATABASE_URL, PYTHONPATH (optional),
#   HF_TELEGRAM_SIGNAL_BOT_TOKEN, HF_TELEGRAM_SIGNAL_CHAT_ID,
#   HF_TELEGRAM_SIGNAL_ENABLED
#
# Idempotent — safe to re-run; existing candidate files get overwritten.
#
# Usage (from root shell that has /etc/neucast/env sourced):
#   set -a; source /etc/neucast/env; set +a
#   sudo --preserve-env=DATABASE_URL,HF_TELEGRAM_SIGNAL_BOT_TOKEN,HF_TELEGRAM_SIGNAL_CHAT_ID,HF_TELEGRAM_SIGNAL_ENABLED \
#        -u stailfx bash /opt/neucast/tools/run_pretrain_finetune_pipeline.sh
#
# NOTE: ``set -e`` intentionally NOT used — fine-tune step failures
# for one symbol shouldn't kill the whole pipeline. The notify step
# at the end runs always so the operator gets feedback either way.

ROOT=/opt/neucast
PRETRAIN_DIR=$ROOT/weights/highfreq/candidate_klines_pretrain
FINETUNE_DIR=$ROOT/weights/highfreq/candidate_finetune
PROD_DIR=$ROOT/weights/highfreq
LOG=/tmp/pipeline.log
PRETRAIN_LOG=/tmp/pretrain.log

mkdir -p "$FINETUNE_DIR"
exec >> "$LOG" 2>&1

echo "================================================================"
echo "[$(date)] pipeline runner started"

# ── Set up Python path UPFRONT so any error-path telegram_notify call
# works even if we exit before the per-symbol fine-tune block.
# The TG / DB env vars are expected to be already in the environment,
# passed by the parent root shell via sudo --preserve-env (we can't
# source /etc/neucast/env ourselves — it's mode 0600 root-only and
# this script runs as stailfx).
cd $ROOT
export PYTHONPATH=$ROOT

# ── Step 1: wait for pretrain to finish ─────────────────────────────
echo "[$(date)] waiting for pretrain process to exit…"
# Check both the bash wrapper script AND the python pretrain module.
# Use two separate pgreps with OR (more robust than regex alternation
# which has different syntax across pgrep / grep -E).
while pgrep -f run_pretrain.sh >/dev/null \
   || pgrep -f "app.highfreq.pretrain" >/dev/null; do
  sleep 30
done
echo "[$(date)] pretrain finished"

# ── Step 2: verify pretrained checkpoints ───────────────────────────
for sym in btcusdt ethusdt bnbusdt; do
  if [ ! -f "$PRETRAIN_DIR/${sym}_1m_pretrained.cbm" ]; then
    echo "[$(date)] ERROR missing pretrained ckpt for $sym"
    /opt/neucast/venv/bin/python -m tools.telegram_notify \
      --text "🚨 NeuCast pretrain pipeline FAILED — missing checkpoint for ${sym^^}"
    exit 1
  fi
done

# ── Step 3: fine-tune each symbol via --init-from ───────────────────
# (cwd / PYTHONPATH already set at script top.)
for sym in BTCUSDT ETHUSDT BNBUSDT; do
  symlower=$(echo $sym | tr A-Z a-z)
  echo "[$(date)] fine-tuning $sym from pretrained ckpt"
  /opt/neucast/venv/bin/python -m app.highfreq.trainer \
    --symbol $sym \
    --since-hours 93 \
    --feature-set long_horizon \
    --init-from $PRETRAIN_DIR/${symlower}_1m_pretrained.cbm \
    --out $FINETUNE_DIR/${symlower}_1m.cbm \
    --report $FINETUNE_DIR/${symlower}_1m_metrics.json \
    --frozen-holdout-days 0 \
    --sample-weight-half-life-bars 0 \
    || echo "[$(date)] WARN fine-tune $sym exited non-zero"
done
echo "[$(date)] all fine-tunes done"

# ── Step 4: comparator ──────────────────────────────────────────────
COMPARE_OUT=/tmp/compare_result.txt
/opt/neucast/venv/bin/python -m tools.compare_candidate_models \
  --production-dir $PROD_DIR \
  --candidate-dir $FINETUNE_DIR \
  --tolerance 0.005 \
  > $COMPARE_OUT 2>&1
COMPARE_RC=$?
echo "[$(date)] comparator exit=$COMPARE_RC"

# ── Step 5: Telegram notification ───────────────────────────────────
{
  echo "🧠 <b>NeuCast HF pretrain → fine-tune complete</b>"
  echo ""
  if [ "$COMPARE_RC" -eq 0 ]; then
    echo "✅ <b>Verdict: deploy</b> (at least one symbol improved, none regressed)"
  elif [ "$COMPARE_RC" -eq 1 ]; then
    echo "⚠️ <b>Verdict: keep production</b> (at least one symbol regressed)"
  else
    echo "❓ <b>Verdict: missing data</b> (no candidate metrics found)"
  fi
  echo ""
  echo "<pre>"
  cat $COMPARE_OUT
  echo "</pre>"
  echo ""
  echo "Pretrain checkpoints in <code>$PRETRAIN_DIR</code>"
  echo "Candidate fine-tunes in <code>$FINETUNE_DIR</code>"
  echo "Run timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > /tmp/tg_body.html

# /etc/neucast/env was already sourced at script top.
/opt/neucast/venv/bin/python -m tools.telegram_notify \
  --from-file /tmp/tg_body.html \
  || echo "[$(date)] telegram notify failed (non-fatal)"

echo "[$(date)] pipeline DONE"
