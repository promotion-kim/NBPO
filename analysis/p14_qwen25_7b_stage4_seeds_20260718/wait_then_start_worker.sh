#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 9 ]] || { echo "usage: $0 PROJECT TRAIN_PY INFER_PY ROOT CACHE GPU LOG PID..." >&2; exit 2; }
PROJECT=$1; TRAIN_PY=$2; INFER_PY=$3; ROOT=$4; CACHE=$5; GPU=$6; LOG=$7; shift 7
for pid in "$@"; do
  while kill -0 "$pid" 2>/dev/null; do sleep 10; done
done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null | grep -q '[0-9]'; do sleep 5; done
exec env CUDA_VISIBLE_DEVICES="$GPU" "$TRAIN_PY" \
  "$PROJECT/analysis/p14_qwen25_7b_stage4_seeds_20260718/worker.py" \
  --project "$PROJECT" --train-python "$TRAIN_PY" --infer-python "$INFER_PY" \
  --root "$ROOT" --cache "$CACHE" --gpu "$GPU" >>"$LOG" 2>&1
