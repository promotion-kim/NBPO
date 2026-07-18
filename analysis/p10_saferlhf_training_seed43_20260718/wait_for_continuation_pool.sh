#!/usr/bin/env bash
# Launch one continuation pool only after its parent is complete and target GPUs are idle.
set -euo pipefail
[[ $# -ge 7 && $# -le 8 ]] || { echo "usage: $0 PROJECT EXP STAGE ARM PARENT GPU_A GPU_B [LOG]" >&2; exit 2; }
PROJECT=$1; EXP=$2; STAGE=$3; ARM=$4; PARENT=$5; GPU_A=$6; GPU_B=$7
LOG=${8:-$EXP/logs/$STAGE/queue_pool_${ARM}_g${GPU_A}_${GPU_B}.log}
mkdir -p "$(dirname "$LOG")"
{
  echo "$(date -Is) queued $STAGE pool for $ARM"
  test -f "$EXP/continuation_lock.json"
  while :; do
    parent_ok=0
    if "$PROJECT/../venv_clean/bin/python" - "$PARENT/job_status.json" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
except (FileNotFoundError,json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if d.get("status") == "completed" and d.get("finite_metrics") else 1)
PY
    then parent_ok=1; fi
    idle=1
    for gpu in "$GPU_A" "$GPU_B"; do
      for _ in 1 2 3; do
        pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
        printf '%s gpu=%s compute_pids=%s\n' "$(date -Is)" "$gpu" "${pids:-none}"
        [[ -z ${pids//[[:space:]]/} ]] || { idle=0; break 2; }
        sleep 3
      done
    done
    [[ $parent_ok -eq 1 && $idle -eq 1 ]] && break
    sleep 20
  done
  echo "$(date -Is) parent and three idle samples passed; building $STAGE pool for $ARM"
  "$PROJECT/analysis/p10_saferlhf_training_seed43_20260718/prepare_continuation_pool.sh" \
    "$PROJECT" "$EXP" "$STAGE" "$ARM" "$PARENT" "$GPU_A" "$GPU_B"
} >>"$LOG" 2>&1
