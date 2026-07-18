#!/usr/bin/env bash
# Queue the reward-blind smoke and final continuation only after a structural pool audit.
set -euo pipefail
[[ $# -eq 10 ]] || { echo "usage: $0 PROJECT VENV EXP STAGE ARM PARENT LOSS TARGET GPU LOG" >&2; exit 2; }
PROJECT=$1; VENV=$2; EXP=$3; STAGE=$4; ARM=$5; PARENT=$6; LOSS=$7; TARGET=$8; GPU=$9; LOG=${10}
POOL=$EXP/$STAGE/$ARM/pool
DATASET=$POOL/precompute/targets/dataset_dict.json
AUDIT=$POOL/POOL_AUDITED.json
mkdir -p "$(dirname "$LOG")"
{
  echo "$(date -Is) queued $STAGE train for $ARM"
  test -f "$EXP/continuation_lock.json"
  while :; do
    parent_ok=0
    if "$VENV/bin/python" - "$PARENT/job_status.json" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
except (FileNotFoundError,json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if d.get("status") == "completed" and d.get("finite_metrics") else 1)
PY
    then parent_ok=1; fi
    if [[ -f "$POOL/PREPARED" && ! -f "$AUDIT" ]]; then
      "$VENV/bin/python" "$PROJECT/analysis/p10_saferlhf_training_seed43_20260718/audit_stage2_pool.py" \
        --pool "$POOL" --expected-prompts 2500 || true
    fi
    idle=1
    for _ in 1 2 3; do
      pids=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
      printf '%s gpu=%s compute_pids=%s\n' "$(date -Is)" "$GPU" "${pids:-none}"
      [[ -z ${pids//[[:space:]]/} ]] || { idle=0; break; }
      sleep 3
    done
    [[ $parent_ok -eq 1 && -f "$DATASET" && -f "$AUDIT" && $idle -eq 1 ]] && break
    sleep 20
  done
  echo "$(date -Is) parent, audited pool, and three idle samples passed; training $STAGE $ARM"
  "$PROJECT/analysis/p10_saferlhf_training_seed43_20260718/run_continuation_arm_sequence.sh" \
    "$PROJECT" "$VENV" "$EXP" "$STAGE" "$ARM" "$PARENT" "$LOSS" "$TARGET" "$GPU" "$LOG"
} >>"$LOG" 2>&1
