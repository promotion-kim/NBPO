#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 4 ]] || exit 2
PROJECT=$1; TRAIN_PY=$2; ROOT=$3; CACHE=$4
TOTAL=$((2*4*8))
mkdir -p "$ROOT/evaluation/logs"
while :; do
  terminal=$(find "$ROOT/scheduler" -maxdepth 1 -type f \( -name '*.DONE.json' -o -name '*.FAILED.json' -o -name '*.BLOCKED.json' \) | wc -l)
  (( terminal >= TOTAL )) && break
  sleep 60
done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; do sleep 30; done
: > "$ROOT/evaluation/logs/prelaunch_gpu_samples.txt"
for sample in 1 2 3; do
  date -Is >> "$ROOT/evaluation/logs/prelaunch_gpu_samples.txt"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> "$ROOT/evaluation/logs/prelaunch_gpu_samples.txt"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader >> "$ROOT/evaluation/logs/prelaunch_gpu_samples.txt"
  (( sample == 3 )) || sleep 4
done
bash "$PROJECT/analysis/p14_qwen25_7b_stage4_seeds_20260718/evaluate_after_training.sh" "$PROJECT" "$TRAIN_PY" "$ROOT" "$CACHE" > "$ROOT/evaluation/logs/evaluate.log" 2>&1
