#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 10 ]] || { echo "usage: $0 PROJECT PY CACHE LOCK PARENT DATASET OUT CAND GPU SOURCE" >&2; exit 2; }
PROJECT=$1 PY=$2 CACHE=$3 LOCK=$4 PARENT=$5 DATASET=$6 OUT=$7 CAND=$8 GPU=$9 SOURCE=${10}
LOG=$OUT/launch/$CAND.log
mkdir -p "$OUT/launch"
"$PY" "$PROJECT/analysis/p17_qwen25_ipo_completion_20260720/train_ipo_stage4_candidate.py" \
  --project "$PROJECT" --python "$PY" --cache "$CACHE" --lock "$LOCK" --parent "$PARENT" \
  --dataset "$DATASET" --output-root "$OUT" --candidate "$CAND" --gpu "$GPU" >> "$LOG" 2>&1
MODEL=$OUT/candidate_runs/$CAND/s43/stage4/train/full
bash "$PROJECT/analysis/p17_qwen25_ipo_completion_20260720/gate_candidate.sh" \
  "$PROJECT" "$PY" "$SOURCE" "$OUT" "$CAND" "$MODEL" "$GPU" "$CACHE" >> "$LOG" 2>&1

