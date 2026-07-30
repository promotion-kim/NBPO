#!/usr/bin/env bash
# Queue reward-blind P8-panel decode/gate for a completed P10 Stage-2 arm.
set -euo pipefail
[[ $# -eq 4 ]] || { echo "usage: $0 PROJECT EXP ARM GPU" >&2; exit 2; }
PROJECT=$1; EXP=$2; ARM=$3; GPU=$4
STATUS=$EXP/stage2/$ARM/train/full/job_status.json
LOG=$EXP/logs/stage2_eval/queue_${ARM}_gpu${GPU}.log
mkdir -p "$(dirname "$LOG")"
{
  echo "$(date -Is) locked-panel Stage-2 evaluation queue registered for $ARM"
  while :; do
    parent_ok=0
    if [[ -s $STATUS ]] && "$PROJECT/../venv_clean/bin/python" - "$STATUS" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
raise SystemExit(0 if d.get("status") == "completed" and d.get("finite_metrics") else 1)
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
  echo "$(date -Is) parent and three idle samples verified; decoding Stage-2 $ARM"
  E="$EXP" bash "$PROJECT/analysis/p10_saferlhf_training_seed43_20260718/decode_and_gate_stage2_arm.sh" "$ARM" "$GPU"
} >>"$LOG" 2>&1
