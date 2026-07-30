#!/usr/bin/env bash
# Start an arm-local Stage-4 run only after its frozen Stage-3 gate passes.
set -euo pipefail
[[ $# -eq 8 ]] || { echo "usage: $0 PROJECT VENV STAGE3_EXP STAGE4_EXP ARM LOSS TARGET GPU" >&2; exit 2; }
PROJECT=$1; VENV=$2; E3=$3; E4=$4; ARM=$5; LOSS=$6; TARGET=$7; GPU=$8
GATE=$E3/stage3_stability_p8_locked_panel/gates/$ARM.json
LOG=$E4/logs/early_stage4/wait_gate_${ARM}_g${GPU}.log
mkdir -p "$(dirname "$LOG")"
{
  echo "$(date -Is) waiting for frozen Stage-3 gate: $ARM"
  while [[ ! -s "$GATE" ]]; do sleep 20; done
  "$VENV/bin/python" - "$GATE" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1], encoding="utf-8"))
if gate.get("status") != "passed" or gate.get("passed") is not True:
    raise SystemExit("Stage-3 gate did not pass; Stage-4 remains fail-closed")
PY
  echo "$(date -Is) Stage-3 gate passed; launching arm-local Stage 4: $ARM"
  exec bash "$PROJECT/analysis/p10_saferlhf_training_seed43_20260718/run_stage4_early_arm.sh" \
    "$PROJECT" "$VENV" "$E3" "$E4" "$ARM" "$LOSS" "$TARGET" "$GPU"
} >>"$LOG" 2>&1
