#!/usr/bin/env bash
# Run one already-gated Stage-3 parent into Stage 4 on an idle authorized GPU.
set -euo pipefail
[[ $# -eq 8 ]] || { echo "usage: $0 PROJECT VENV STAGE3_EXP STAGE4_EXP ARM LOSS TARGET GPU" >&2; exit 2; }
PROJECT=$1; VENV=$2; E3=$3; E4=$4; ARM=$5; LOSS=$6; TARGET=$7; GPU=$8
R=$PROJECT/analysis/p10_saferlhf_training_seed43_20260718
PARENT=$E3/stage3/$ARM/train/full
GATE=$E3/stage3_stability_p8_locked_panel/gates/$ARM.json
LOCK=$E4/early_locks/$ARM.json
LOG=$E4/logs/early_stage4/${ARM}_g${GPU}.log
mkdir -p "$(dirname "$LOG")"
{
  echo "$(date -Is) early Stage-4 arm registration: $ARM gpu=$GPU"
  "$VENV/bin/python" "$R/write_early_stage4_lock.py" --output "$LOCK" --arm "$ARM" \
    --loss-type "$LOSS" --target-column "$TARGET" --parent "$PARENT" --parent-gate "$GATE"
  for _ in 1 2 3; do
    pids=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
    echo "$(date -Is) prelaunch gpu=$GPU compute_pids=${pids:-none}"
    [[ -z ${pids//[[:space:]]/} ]] || { echo "GPU became busy; fail closed before pool"; exit 1; }
    sleep 3
  done
  "$R/prepare_continuation_pool.sh" "$PROJECT" "$E4" stage4 "$ARM" "$PARENT" "$GPU" "$GPU"
  "$VENV/bin/python" "$R/audit_stage2_pool.py" --pool "$E4/stage4/$ARM/pool" --expected-prompts 2500
  "$R/run_continuation_arm_sequence.sh" "$PROJECT" "$VENV" "$E4" stage4 "$ARM" "$PARENT" "$LOSS" "$TARGET" "$GPU" "$LOG"
  while :; do
    idle=1
    for _ in 1 2 3; do
      pids=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
      [[ -z ${pids//[[:space:]]/} ]] || { idle=0; break; }
      sleep 3
    done
    [[ $idle -eq 1 ]] && break
    sleep 20
  done
  "$R/decode_and_gate_continuation.sh" "$PROJECT" "$E4" stage4 "$ARM" "$GPU"
  echo "$(date -Is) early Stage-4 arm terminal: $ARM"
} >>"$LOG" 2>&1
