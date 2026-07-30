#!/usr/bin/env bash
# Queue one locked Stage-2 per-method pool after its Stage-1 parent completes.
set -euo pipefail
[[ $# -eq 5 ]] || { echo "usage: $0 PROJECT EXP ARM GPU_A GPU_B" >&2; exit 2; }
PROJECT=$1; EXP=$2; ARM=$3; GPU_A=$4; GPU_B=$5
PARENT=$EXP/stage1/$ARM/train/full
STATUS_DIR=$PARENT
LOG=$EXP/logs/stage2/queue_pool_${ARM}_g${GPU_A}_${GPU_B}.log
mkdir -p "$(dirname "$LOG")"
{
  echo "$(date -Is) Stage-2 pool queue registered for $ARM"
  while :; do
    parent_ok=0
    if "$PROJECT/../venv_clean/bin/python" - "$STATUS_DIR" <<'PY'
import json,sys
root=sys.argv[1]
for name in ("job_status_repaired.json", "job_status.json"):
    path=f"{root}/{name}"
    try:
        d=json.load(open(path))
    except FileNotFoundError:
        continue
    if d.get("status") == "completed" and d.get("finite_metrics"):
        raise SystemExit(0)
raise SystemExit(1)
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
  echo "$(date -Is) parent completed and three idle samples passed; preparing Stage-2 pool for $ARM"
  # This only changes vLLM cache reservation, not sampling or training.  The
  # previous 0.88 reservation OOMed while the two parent engines initialized.
  VLLM_GPU_MEMORY_UTILIZATION=0.55 PROJECT="$PROJECT" bash "$PROJECT/analysis/p5_8b_robust_stage1_stage2_20260717/prepare_stage2_pool.sh" "$EXP" "$ARM" "$PARENT" "$GPU_A" "$GPU_B"
} >>"$LOG" 2>&1
