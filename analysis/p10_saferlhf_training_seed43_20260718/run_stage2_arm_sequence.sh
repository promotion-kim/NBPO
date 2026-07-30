#!/usr/bin/env bash
# Launch a locked P10 Stage-2 smoke then full run after its pool is ready.
set -euo pipefail
[[ $# -eq 9 ]] || { echo "usage: $0 PROJECT VENV EXP BASE ARM PARENT LOSS TARGET GPU" >&2; exit 2; }
PROJECT=$1; VENV=$2; EXP=$3; BASE=$4; ARM=$5; PARENT=$6; LOSS=$7; TARGET=$8; GPU=$9
[[ -n ${WANDB_API_KEY:-} ]] || { echo "WANDB_API_KEY required" >&2; exit 2; }
RUNNER=$PROJECT/analysis/p10_saferlhf_training_seed43_20260718/train_stage2_arm.py
COMMON=(--project "$PROJECT" --venv "$VENV" --experiment "$EXP" --arm "$ARM" --parent-model "$PARENT" --loss-type "$LOSS" --gpu "$GPU" --seed "${TRAIN_SEED:-43}" --run-prefix "${RUN_PREFIX:-p10}")
[[ $TARGET != - ]] && COMMON+=(--target-column "$TARGET")
"$VENV/bin/python" "$RUNNER" "${COMMON[@]}" --stage smoke
"$VENV/bin/python" "$RUNNER" "${COMMON[@]}" --stage full
