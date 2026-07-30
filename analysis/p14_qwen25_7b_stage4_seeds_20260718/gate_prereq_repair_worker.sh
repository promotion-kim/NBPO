#!/usr/bin/env bash
set -u
[[ $# -ge 6 ]] || { echo "usage: $0 PROJECT TRAIN_PY INFER_PY ROOT GPU seed:arm..." >&2; exit 2; }
PROJECT=$1; TRAIN_PY=$2; INFER_PY=$3; ROOT=$4; GPU=$5; shift 5
for spec in "$@"; do
  seed=${spec%%:*}; arm=${spec#*:}
  bash "$PROJECT/analysis/p14_qwen25_7b_stage4_seeds_20260718/decode_and_gate.sh" \
    "$PROJECT" "$TRAIN_PY" "$INFER_PY" "$ROOT" "$seed" 1 "$arm" "$GPU" || true
done
