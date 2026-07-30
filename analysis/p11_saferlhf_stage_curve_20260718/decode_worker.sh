#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 5 ]] || { echo "usage: $0 PROJECT OUTPUT GPU DONE_MARKER MODEL_KEY [...]" >&2; exit 2; }
PROJECT=$1; OUT=$2; GPU=$3; DONE=$4; shift 4
DECODER=$PROJECT/analysis/p11_saferlhf_stage_curve_20260718/decode_and_gate.sh
for key in "$@"; do
  model=$(awk -F '\t' -v key="$key" '$1==key {print $4}' "$OUT/models.tsv")
  [[ -n "$model" ]] || { echo "unknown model key: $key" >&2; exit 1; }
  "$DECODER" "$PROJECT" "$OUT" "$key" "$model" "$GPU"
done
date -Is > "$DONE"
