#!/usr/bin/env bash
# Decode a fixed list of diagnostic-only models, one worker per local GPU.
# Task strings are NAME=ABSOLUTE_MODEL_PATH.  Workers process their assigned
# tasks sequentially so no GPU receives two concurrent vLLM engines.
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 GPU_CSV MANIFEST OUTPUT_ROOT NAME=MODEL [NAME=MODEL ...]" >&2
  exit 2
fi
GPUS_CSV=$1
MANIFEST=$2
OUT=$3
shift 3
TASKS=("$@")

PROJECT=${PROJECT:-/NHNHOME/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/AIPR/sjkim/venv_clean}
IFS=, read -r -a GPUS <<< "$GPUS_CSV"
[[ ${#GPUS[@]} -gt 0 ]] || { echo "no GPUs supplied" >&2; exit 2; }
EXPECTED=$(wc -l < "$MANIFEST")
[[ "$EXPECTED" == "693" ]] || { echo "unexpected retrospective manifest rows: $EXPECTED" >&2; exit 1; }
mkdir -p "$OUT/generations" "$OUT/logs"

run_one() {
  local gpu=$1 task=$2 name model output
  name=${task%%=*}
  model=${task#*=}
  [[ "$name" != "$model" && -d "$model" ]] || { echo "invalid task: $task" >&2; return 2; }
  output="$OUT/generations/$name/output_42.json"
  mkdir -p "$(dirname "$output")"
  CUDA_VISIBLE_DEVICES=$gpu \
    PYTHONPATH="$PROJECT" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    TOKENIZERS_PARALLELISM=false \
    TORCH_CUDNN_SDPA_ENABLED=0 \
    MNPO_DISABLE_CUDNN_SDPA=1 \
    "$VENV/bin/python" "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py" \
      --manifest "$MANIFEST" --model "$model" --policy-name "$name" --probe none --output "$output" \
      --seed 42 --temperature 0.7 --top-p 0.9 --max-new-tokens 512 --max-model-len 8192 --gpu-memory-utilization 0.88 \
      > "$OUT/logs/decode_${name}.log" 2>&1
}

pids=()
for gpu_index in "${!GPUS[@]}"; do
  gpu=${GPUS[$gpu_index]}
  (
    for task_index in "${!TASKS[@]}"; do
      if (( task_index % ${#GPUS[@]} == gpu_index )); then
        run_one "$gpu" "${TASKS[$task_index]}"
      fi
    done
  ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
date -Is > "$OUT/DECODE_COMPLETE_$(hostname)"
