#!/usr/bin/env bash
# Queue a locked Stage-1 arm behind the current job on one authorized GPU.
set -euo pipefail
[[ $# -eq 8 ]] || { echo "usage: $0 PROJECT VENV EXP BASE ARM LOSS TARGET GPU" >&2; exit 2; }
PROJECT=$1; VENV=$2; EXP=$3; BASE=$4; ARM=$5; LOSS=$6; TARGET=$7; GPU=$8
[[ -n ${WANDB_API_KEY:-} ]] || { echo "WANDB_API_KEY required" >&2; exit 2; }
LOG=$EXP/logs/stage1/queue_${ARM}_gpu${GPU}.log
mkdir -p "$(dirname "$LOG")"
{
  echo "$(date -Is) queue registered for $ARM on GPU $GPU"
  while :; do
    idle=1
    for _ in 1 2 3; do
      snapshot=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
      printf '%s gpu=%s compute_pids=%s\n' "$(date -Is)" "$GPU" "${snapshot:-none}"
      [[ -z ${snapshot//[[:space:]]/} ]] || { idle=0; break; }
      sleep 3
    done
    [[ $idle -eq 1 ]] && break
    sleep 15
  done
  echo "$(date -Is) three idle read-only samples passed; launching $ARM"
  "$PROJECT/analysis/p10_saferlhf_training_seed43_20260718/run_stage1_arm_sequence.sh" "$PROJECT" "$VENV" "$EXP" "$BASE" "$ARM" "$LOSS" "$TARGET" "$GPU"
} >>"$LOG" 2>&1
