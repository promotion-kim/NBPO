#!/usr/bin/env bash
# Serialize independent Stage-2 pool builds on one authorized GPU pair.
set -euo pipefail
[[ $# -eq 6 ]] || { echo "usage: $0 PROJECT EXP PREREQUISITE ARM GPU_A GPU_B" >&2; exit 2; }
PROJECT=$1; EXP=$2; PREREQUISITE=$3; ARM=$4; GPU_A=$5; GPU_B=$6
LOG=$EXP/logs/stage2/queue_after_$(basename "$PREREQUISITE")_${ARM}_g${GPU_A}_${GPU_B}.log
mkdir -p "$(dirname "$LOG")"
{
  echo "$(date -Is) queued $ARM Stage-2 pool after $PREREQUISITE"
  while [[ ! -f $PREREQUISITE ]]; do sleep 20; done
  echo "$(date -Is) prerequisite ready; handing off to safe pool launcher for $ARM"
  "$PROJECT/analysis/p10_saferlhf_training_seed43_20260718/wait_for_idle_then_prepare_stage2.sh" \
    "$PROJECT" "$EXP" "$ARM" "$GPU_A" "$GPU_B"
} >>"$LOG" 2>&1
