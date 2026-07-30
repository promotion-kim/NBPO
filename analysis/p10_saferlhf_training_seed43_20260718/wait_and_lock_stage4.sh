#!/usr/bin/env bash
# Create the Stage-4 lock only after every Stage-3 final model has passed the fixed stability gate.
set -euo pipefail
[[ $# -eq 3 ]] || { echo "usage: $0 PROJECT STAGE3_EXPERIMENT STAGE4_EXPERIMENT" >&2; exit 2; }
PROJECT=$1; E3=$2; E4=$3
VENV=$PROJECT/../venv_clean
RUNNER=$PROJECT/analysis/p10_saferlhf_training_seed43_20260718/write_continuation_lock.py
ARMS=(ronpo_os ronpo_topmass inpo_avg sppo_avg simpo ipo dpo ht_mnpo_harmless ht_mnpo_helpfulness)
LOG=$E3/logs/stage4_lock_queue.log
mkdir -p "$(dirname "$LOG")"
{
  echo "$(date -Is) Stage-4 lock queue registered"
  while :; do
    ready=1
    for arm in "${ARMS[@]}"; do
      gate=$E3/stage3_stability_p8_locked_panel/gates/$arm.json
      "$VENV/bin/python" - "$gate" <<'PY' || { ready=0; break; }
import json,sys
try:
    d=json.load(open(sys.argv[1]))
except (FileNotFoundError,json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if d.get("status") == "passed" and d.get("passed") is True else 1)
PY
    done
    [[ $ready -eq 1 ]] && break
    sleep 30
  done
  [[ ! -e "$E4/continuation_lock.json" ]] || { echo "existing Stage-4 lock; refusing duplicate" >&2; exit 1; }
  mkdir -p "$E4"
  "$VENV/bin/python" "$RUNNER" --output "$E4/continuation_lock.json" --stage stage4 \
    --parent-experiment "$E3" --parent-stage stage3 \
    --required-gates "$E3/stage3_stability_p8_locked_panel"
  echo "$(date -Is) Stage-4 lock created"
} >>"$LOG" 2>&1
