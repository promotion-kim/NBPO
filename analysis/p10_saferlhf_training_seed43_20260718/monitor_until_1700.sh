#!/usr/bin/env bash
# Read-only, periodic P10 evidence capture. It never starts, stops, or edits jobs.
set -euo pipefail
[[ $# -eq 4 ]] || { echo "usage: $0 PROJECT VENV EXP HOST" >&2; exit 2; }
PROJECT=$1; VENV=$2; EXP=$3; HOST=$4
DEADLINE=$(date -d '2026-07-18 17:00:00 +0900' +%s)
while :; do
  "$VENV/bin/python" "$PROJECT/analysis/p10_saferlhf_training_seed43_20260718/write_snapshot.py" \
    --experiment "$EXP" --host "$HOST" --stage "seed43_stage1_stage2_continuation" || true
  [[ $(date +%s) -ge $DEADLINE ]] && break
  sleep 1800
done
