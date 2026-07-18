#!/usr/bin/env bash
# Queue fixed-panel decode and reward-blind stability gate for a completed P10 arm.
set -euo pipefail
[[ $# -eq 4 ]] || { echo "usage: $0 PROJECT EXP ARM GPU" >&2; exit 2; }
PROJECT=$1; EXP=$2; ARM=$3; GPU=$4
STATUS_DIR=$EXP/stage1/$ARM/train/full
LOG=$EXP/logs/stage1_eval/queue_${ARM}_gpu${GPU}.log
mkdir -p "$(dirname "$LOG")"
{
  echo "$(date -Is) fixed-panel evaluation queue registered for $ARM"
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
    for _ in 1 2 3; do
      pids=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
      printf '%s gpu=%s compute_pids=%s\n' "$(date -Is)" "$GPU" "${pids:-none}"
      [[ -z ${pids//[[:space:]]/} ]] || { idle=0; break; }
      sleep 3
    done
    [[ $parent_ok -eq 1 && $idle -eq 1 ]] && break
    sleep 20
  done
  echo "$(date -Is) parent completed and three idle samples passed; decoding $ARM"
  E="$EXP" bash "$PROJECT/analysis/p10_saferlhf_training_seed43_20260718/decode_and_gate_stage1_arm.sh" "$ARM" "$GPU"
} >>"$LOG" 2>&1
