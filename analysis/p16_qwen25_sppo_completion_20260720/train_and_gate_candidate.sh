#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 12 ]] || { echo "usage: $0 PROJECT PY CACHE LOCK PARENT DATASET OUTPUT CANDIDATE SEED GPU SOURCE LOG" >&2; exit 2; }
PROJECT=$1 PY=$2 CACHE=$3 LOCK=$4 PARENT=$5 DATASET=$6 OUT=$7 CAND=$8 SEED=$9 GPU=${10} SOURCE=${11} LOG=${12}
"$PY" "$PROJECT/analysis/p16_qwen25_sppo_completion_20260720/train_sppo_stage4_candidate.py" \
  --project "$PROJECT" --python "$PY" --cache "$CACHE" --lock "$LOCK" --parent "$PARENT" \
  --dataset "$DATASET" --output-root "$OUT" --candidate "$CAND" --seed "$SEED" --gpu "$GPU" >> "$LOG" 2>&1
MODEL=$OUT/$CAND/s$SEED/stage4/train/full
bash "$PROJECT/analysis/p16_qwen25_sppo_completion_20260720/gate_sppo_stage4_candidate.sh" \
  "$PROJECT" "$PY" "$SOURCE" "$OUT" "$CAND" "$SEED" "$MODEL" "$GPU" "$CACHE" >> "$LOG" 2>&1

