#!/usr/bin/env bash
# Queue an unchanged reward-blind stability gate after an arm finishes training.
set -euo pipefail
[[ $# -eq 6 ]] || { echo "usage: $0 PROJECT EXP STAGE ARM GPU LOG" >&2; exit 2; }
PROJECT=$1; EXP=$2; STAGE=$3; ARM=$4; GPU=$5; LOG=$6
STATUS=$EXP/$STAGE/$ARM/train/full/job_status.json
mkdir -p "$(dirname "$LOG")"
{
  echo "$(date -Is) queued $STAGE stability gate for $ARM"
  while :; do
    complete=0
    if "$PROJECT/../venv_clean/bin/python" - "$STATUS" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
except (FileNotFoundError,json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if d.get("status") == "completed" and d.get("finite_metrics") else 1)
PY
    then complete=1; fi
    idle=1
    for _ in 1 2 3; do
      pids=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
      printf '%s gpu=%s compute_pids=%s\n' "$(date -Is)" "$GPU" "${pids:-none}"
      [[ -z ${pids//[[:space:]]/} ]] || { idle=0; break; }
      sleep 3
    done
    [[ $complete -eq 1 && $idle -eq 1 ]] && break
    sleep 20
  done
  echo "$(date -Is) training and three idle samples passed; decoding/gating $STAGE $ARM"
  "$PROJECT/analysis/p10_saferlhf_training_seed43_20260718/decode_and_gate_continuation.sh" \
    "$PROJECT" "$EXP" "$STAGE" "$ARM" "$GPU"
} >>"$LOG" 2>&1
