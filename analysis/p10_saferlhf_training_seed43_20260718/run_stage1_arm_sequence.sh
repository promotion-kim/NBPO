#!/usr/bin/env bash
# Launch wrapper. The caller supplies WANDB_API_KEY only through its process environment.
set -euo pipefail

if [ "$#" -ne 8 ]; then
  echo "usage: $0 PROJECT VENV EXPERIMENT BASE ARM LOSS_TYPE TARGET_COLUMN GPU" >&2
  exit 2
fi

PROJECT=$1
VENV=$2
EXPERIMENT=$3
BASE=$4
ARM=$5
LOSS_TYPE=$6
TARGET_COLUMN=$7
GPU=$8

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WANDB_API_KEY is required in the inherited environment" >&2
  exit 2
fi

RUNNER="$PROJECT/analysis/p10_saferlhf_training_seed43_20260718/train_stage1_arm.py"
COMMON=(
  --project "$PROJECT" --venv "$VENV" --experiment "$EXPERIMENT" --base "$BASE"
  --arm "$ARM" --loss-type "$LOSS_TYPE" --gpu "$GPU" --seed "${TRAIN_SEED:-43}" --run-prefix "${RUN_PREFIX:-p10}"
)
if [ "$TARGET_COLUMN" != "-" ]; then
  COMMON+=(--target-column "$TARGET_COLUMN")
fi

"$VENV/bin/python" "$RUNNER" "${COMMON[@]}" --stage smoke
"$VENV/bin/python" "$RUNNER" "${COMMON[@]}" --stage full
