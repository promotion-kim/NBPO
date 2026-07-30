#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 9 ]] || { echo "usage: $0 PROJECT PY SOURCE OUTPUT_ROOT CANDIDATE SEED GPU CACHE LOG" >&2; exit 2; }
PROJECT=$1 PY=$2 SOURCE=$3 OUT=$4 CAND=$5 SEED=$6 GPU=$7 CACHE=$8 LOG=$9
STATUS=$OUT/$CAND/s$SEED/stage4/train/full/job_status.json
while [[ ! -s $STATUS ]]; do sleep 20; done
[[ $("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status"))' "$STATUS") == completed ]]
MODEL=$OUT/$CAND/s$SEED/stage4/train/full
bash "$PROJECT/analysis/p16_qwen25_sppo_completion_20260720/gate_sppo_stage4_candidate.sh" \
  "$PROJECT" "$PY" "$SOURCE" "$OUT" "$CAND" "$SEED" "$MODEL" "$GPU" "$CACHE" > "$LOG" 2>&1

