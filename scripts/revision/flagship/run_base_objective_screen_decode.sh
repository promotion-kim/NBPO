#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/venv_clean}
EXP=$PROJECT/results/p1_8b_base_objective_screen_20260716
CACHE=${CACHE:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/revision_qwen3_8b/full_iter1/base_objective_screen/cache}
DOWNLOADS=$EXP/downloads
MANIFEST=$EXP/dataset_manifest/validation.jsonl
DECODER=$PROJECT/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py
mkdir -p "$EXP/generations" "$EXP/logs/decode" "$EXP/decode_status"
source "$VENV/bin/activate"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_CUDNN_SDPA_ENABLED=0
export WANDB_MODE=disabled

snapshot() {
  local name=$1
  local path_file=$DOWNLOADS/$name.path
  [[ -s "$path_file" ]] || { echo "missing snapshot path: $path_file" >&2; return 1; }
  tail -1 "$path_file" | sed -n 's/^  path: //p; t; p'
}

LLAMA=$(snapshot llama31)
QWEN=$(snapshot qwen25)
MISTRAL=$(snapshot mistral7)
ZEPHYR=$(snapshot zephyr)
WEAK=/NHNHOME/26msit001_A/BASE/AIPR/sjkim/revision_qwen3_8b/full_iter1/flagship_20260712/cache/huggingface/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306

run_one() {
  local gpu=$1 name=$2 model=$3 probe=$4
  local output=$EXP/generations/$name/output_42.json
  if [[ -f "$output" ]] && [[ $(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$output") -eq 640 ]]; then
    printf '%s\t%s\tcached\n' "$(date -Is)" "$name" >"$EXP/decode_status/$name.status"
    return 0
  fi
  mkdir -p "$EXP/generations/$name"
  printf '%s\t%s\trunning\tgpu=%s\n' "$(date -Is)" "$name" "$gpu" >"$EXP/decode_status/$name.status"
  if CUDA_VISIBLE_DEVICES=$gpu python "$DECODER" \
      --manifest "$MANIFEST" --model "$model" --policy-name "$name" --probe "$probe" \
      --output "$output" --seed 42 --temperature 0.7 --top-p 0.9 --max-new-tokens 1024 \
      >"$EXP/logs/decode/$name.log" 2>&1; then
    printf '%s\t%s\tcomplete\tgpu=%s\n' "$(date -Is)" "$name" "$gpu" >"$EXP/decode_status/$name.status"
  else
    code=$?; printf '%s\t%s\tfailed_exit_%s\tgpu=%s\n' "$(date -Is)" "$name" "$code" "$gpu" >"$EXP/decode_status/$name.status"; return "$code"
  fi
}

worker0() { run_one 0 llama31 "$LLAMA" none; run_one 0 llama31_over_refusing "$LLAMA" over_refusing; run_one 0 llama31_terse "$LLAMA" terse; }
worker1() { run_one 1 qwen25 "$QWEN" none; run_one 1 qwen25_over_refusing "$QWEN" over_refusing; run_one 1 qwen25_terse "$QWEN" terse; }
worker2() { run_one 2 mistral7 "$MISTRAL" none; run_one 2 mistral7_over_refusing "$MISTRAL" over_refusing; run_one 2 mistral7_terse "$MISTRAL" terse; }
worker3() { run_one 3 unsafe_zephyr "$ZEPHYR" none; run_one 3 weak_small "$WEAK" none; }

worker0 & p0=$!; worker1 & p1=$!; worker2 & p2=$!; worker3 & p3=$!
wait "$p0" "$p1" "$p2" "$p3"
python "$PROJECT/analysis/qwen3_8b_base_objective_screen_20260716/merge_generation_pool.py" \
  --generation-root "$EXP/generations" --output "$EXP/response_pool.jsonl" \
  --diagnostics "$EXP/generation_diagnostics.json" >"$EXP/logs/merge.log" 2>&1
date -Is >"$EXP/generations/DECODE_COMPLETE"
