#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 4 ]] || { echo "usage: $0 PROJECT VENV OUTPUT EXPECTED" >&2; exit 2; }
while :; do
  count=$(find "$3" -name result.json -type f | wc -l)
  [[ $count -ge $4 ]] && break
  sleep 30
done
"$2/bin/python" "$1/analysis/p10_saferlhf_training_seed43_20260718/finalize_p8_stage4_mmlu.py" --output "$3"
