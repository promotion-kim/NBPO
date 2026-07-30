#!/usr/bin/env bash
# Queue raw Beaver scoring for an existing, gate-passing P10 Stage-1 decode.
set -euo pipefail
[[ $# -eq 4 ]] || { echo "usage: $0 PROJECT EXP ARM GPU" >&2; exit 2; }
PROJECT=$1; EXP=$2; ARM=$3; GPU=$4
GATE=$EXP/stage1_eval_p8_locked_panel/gates/$ARM.json
OUT=$EXP/stage1_eval_p8_locked_panel/scores_individual/$ARM
LOG=$EXP/logs/stage1_eval/queue_score_${ARM}_gpu${GPU}.log
mkdir -p "$(dirname "$LOG")"
{
  echo "$(date -Is) raw-score queue registered for $ARM"
  while :; do
    gate_ok=0
    if [[ -s $GATE ]] && "$PROJECT/../venv_clean/bin/python" - "$GATE" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
raise SystemExit(0 if d.get("passed") is True else 1)
PY
    then gate_ok=1; fi
    [[ -s $OUT/helpfulness.jsonl && -s $OUT/harmlessness.jsonl ]] && { echo "$(date -Is) scores already complete"; exit 0; }
    idle=1
    for _ in 1 2 3; do
      pids=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
      printf '%s gpu=%s compute_pids=%s\n' "$(date -Is)" "$GPU" "${pids:-none}"
      [[ -z ${pids//[[:space:]]/} ]] || { idle=0; break; }
      sleep 3
    done
    [[ $gate_ok -eq 1 && $idle -eq 1 ]] && break
    sleep 20
  done
  echo "$(date -Is) gate and three idle samples verified; scoring $ARM"
  E="$EXP" bash "$PROJECT/analysis/p10_saferlhf_training_seed43_20260718/score_stage1_policy_beaver.sh" "$ARM" "$GPU"
} >>"$LOG" 2>&1
