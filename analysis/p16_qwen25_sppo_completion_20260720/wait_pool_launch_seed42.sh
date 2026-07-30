#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 6 ]] || { echo "usage: $0 PROJECT PY CACHE SOURCE REPAIR_ROOT OUTPUT_ROOT" >&2; exit 2; }
PROJECT=$1 PY=$2 CACHE=$3 SOURCE=$4 REPAIR=$5 OUT=$6
POOL=$REPAIR/seeds/s42/stage4/sppo_avg/pool
while [[ ! -s $POOL/PREPARED ]]; do sleep 20; done
LOCK=$PROJECT/analysis/p16_qwen25_sppo_completion_20260720/repair_lock.json
PARENT=$REPAIR/seeds/s42/stage3/sppo_avg/train/full
DATASET=$POOL/precompute/targets
mkdir -p "$OUT/launch"
pids=()
gpu=0
for candidate in sppo_strong_a sppo_strong_b sppo_strong_c; do
  bash "$PROJECT/analysis/p16_qwen25_sppo_completion_20260720/train_and_gate_candidate.sh" \
    "$PROJECT" "$PY" "$CACHE" "$LOCK" "$PARENT" "$DATASET" "$OUT/candidate_runs" \
    "$candidate" 42 "$gpu" "$SOURCE" "$OUT/launch/seed42_${candidate}.log" &
  pids+=("$!")
  gpu=$((gpu + 1))
done
for pid in "${pids[@]}"; do wait "$pid" || true; done
date -Is > "$OUT/seed42_candidates_complete"

