#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/NHNHOME/WORKSPACE/26msit001_A/mnpo"
EXP_ROOT="$BASE_DIR/revision_qwen3_8b/full_iter1"
LOCAL_PATH="$EXP_ROOT/train/ronpo_full_expect_s42_nhn-b200-q3-ronpo-full-s42-07101256"
LEDGER="$EXP_ROOT/hf_uploads/b200_revision_queue_uploads.jsonl"
LOG="$EXP_ROOT/logs/hf_upload_ronpo_full_manual_20260710.log"

set -a
# shellcheck disable=SC1091
source "$BASE_DIR/.secrets/hf.env"
set +a

"$BASE_DIR/venv_clean/bin/python" \
  "$BASE_DIR/MNPO_rev_20260708/scripts/revision/upload_checkpoint_to_hf.py" \
  --local-path "$LOCAL_PATH" \
  --repo-id "promotion/qwen3-8b-ronpo-full-expect-s42" \
  --method "RONPO full expectation" \
  --seed 42 \
  --notes "Qwen3-8B full atom expectation revision run; final step 437." \
  --ledger "$LEDGER" \
  2>&1 | tee -a "$LOG"
