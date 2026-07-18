#!/usr/bin/env bash
# Queue one locked P10 Stage-2 arm after its own Stage-2 pool is ready.
# The launcher must supply WANDB_API_KEY; this script never persists it.
set -euo pipefail
[[ $# -eq 9 ]] || { echo "usage: $0 PROJECT VENV EXP BASE ARM PARENT LOSS TARGET GPU" >&2; exit 2; }
PROJECT=$1; VENV=$2; EXP=$3; BASE=$4; ARM=$5; PARENT=$6; LOSS=$7; TARGET=$8; GPU=$9
[[ -n ${WANDB_API_KEY:-} ]] || { echo "WANDB_API_KEY required" >&2; exit 2; }
DATASET=$EXP/stage2/$ARM/pool/precompute/targets/dataset_dict.json
POOL=$EXP/stage2/$ARM/pool
POOL_AUDIT=$POOL/POOL_AUDITED.json
STATUS=$EXP/stage1/$ARM/train/full/job_status.json
LOG=$EXP/logs/stage2/queue_train_${ARM}_gpu${GPU}.log
mkdir -p "$(dirname "$LOG")"
{
  echo "$(date -Is) Stage-2 train queue registered for $ARM on GPU $GPU"
  while :; do
    parent_ok=0
    if [[ -s $STATUS ]] && "$VENV/bin/python" - "$STATUS" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
raise SystemExit(0 if d.get("status") == "completed" and d.get("finite_metrics") else 1)
PY
    then parent_ok=1; fi
    if [[ -f $POOL/PREPARED ]]; then
      audit_stale=0
      if [[ ! -f $POOL_AUDIT ]]; then
        audit_stale=1
      else
        for artifact in "$POOL/PREPARED" "$POOL/response_pool.jsonl" \
          "$POOL/scores/helpfulness.jsonl" "$POOL/scores/harmlessness.jsonl" \
          "$POOL/pairs_train.jsonl" "$POOL/pairs_test.jsonl" \
          "$POOL/precompute/targets/dataset_dict.json" "$POOL/pool_audit.json"; do
          [[ $artifact -nt $POOL_AUDIT ]] && { audit_stale=1; break; }
        done
      fi
      if [[ $audit_stale -eq 1 ]]; then
        rm -f "$POOL_AUDIT"
        if ! "$VENV/bin/python" "$PROJECT/analysis/p10_saferlhf_training_seed43_20260718/audit_stage2_pool.py" \
          --pool "$POOL" --expected-prompts 2500; then
          printf '%s pool audit pending for %s\n' "$(date -Is)" "$ARM"
          sleep 20
          continue
        fi
      fi
    fi
    idle=1
    for _ in 1 2 3; do
      pids=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
      printf '%s gpu=%s compute_pids=%s\n' "$(date -Is)" "$GPU" "${pids:-none}"
      [[ -z ${pids//[[:space:]]/} ]] || { idle=0; break; }
      sleep 3
    done
    [[ $parent_ok -eq 1 && -f $DATASET && -f $POOL_AUDIT && $idle -eq 1 ]] && break
    sleep 20
  done
  echo "$(date -Is) parent, pool, and three idle samples verified; starting Stage-2 $ARM"
  "$PROJECT/analysis/p10_saferlhf_training_seed43_20260718/run_stage2_arm_sequence.sh" \
    "$PROJECT" "$VENV" "$EXP" "$BASE" "$ARM" "$PARENT" "$LOSS" "$TARGET" "$GPU"
} >>"$LOG" 2>&1
