#!/usr/bin/env bash
# Run a reward-blind smoke and a final matched-budget continuation on one authorized GPU.
set -euo pipefail
[[ $# -eq 10 ]] || { echo "usage: $0 PROJECT VENV EXP STAGE ARM PARENT LOSS TARGET GPU LOG" >&2; exit 2; }
PROJECT=$1; VENV=$2; EXP=$3; STAGE=$4; ARM=$5; PARENT=$6; LOSS=$7; TARGET=$8; GPU=$9; LOG=${10}
RUNNER=$PROJECT/analysis/p10_saferlhf_training_seed43_20260718/train_continuation_arm.py
COMMON=(--project "$PROJECT" --venv "$VENV" --experiment "$EXP" --continuation-stage "$STAGE" --arm "$ARM" --parent-model "$PARENT" --loss-type "$LOSS" --gpu "$GPU" --seed "${TRAIN_SEED:-43}" --run-prefix "${RUN_PREFIX:-p10}")
[[ "$TARGET" != - ]] && COMMON+=(--target-column "$TARGET")
"$VENV/bin/python" "$RUNNER" "${COMMON[@]}" --run-stage smoke >> "$LOG" 2>&1
"$VENV/bin/python" "$RUNNER" "${COMMON[@]}" --run-stage full >> "$LOG" 2>&1
