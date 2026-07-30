#!/usr/bin/env bash
# Finalize a fixed ARC-Challenge cohort only after all locked results exist.
set -euo pipefail
[[ $# -eq 4 ]] || { echo "usage: $0 PROJECT VENV OUTPUT EXPECTED_RESULTS" >&2; exit 2; }
PROJECT=$1
VENV=$2
OUTPUT=$3
EXPECTED=$4
while :; do
  count=$(find "$OUTPUT" -name result.json -type f | wc -l)
  [[ $count -ge $EXPECTED ]] && break
  sleep 30
done
"$VENV/bin/python" "$PROJECT/analysis/p10_saferlhf_training_seed43_20260718/finalize_p8_stage4_arc_challenge.py" --output "$OUTPUT"
