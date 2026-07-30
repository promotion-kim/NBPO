#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/venv_clean}
EXP=$PROJECT/results/p2_8b_hh_multiobjective_20260717
ROOT=/NHNHOME/26msit001_A/BASE/AIPR/sjkim/revision_qwen3_8b/full_iter1
BASE=$ROOT/base_objective_screen/hf_ipv4/llama31
MANIFEST=$PROJECT/results/p1_8b_base_objective_screen_20260716/dataset_manifest/validation.jsonl
EXPECTED_SHA=1d21a88603095af464fa64a057736c4717ec4ac69a377ffe1720378f19319ce9
ER=$EXP/validation

[[ -f "$EXP/train/stretch/full/ipo/checkpoint-900/config.json" ]]
[[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" == "$EXPECTED_SHA" ]]
mkdir -p "$ER/generations" "$ER/logs"
source "$VENV/bin/activate"
export PYTHONPATH=$PROJECT
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_CUDNN_SDPA_ENABLED=0
export WANDB_MODE=disabled

: >"$ER/logs/predecode_gpu_samples.txt"
for sample in 1 2 3; do
  date -Is >>"$ER/logs/predecode_gpu_samples.txt"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader >>"$ER/logs/predecode_gpu_samples.txt"
  for gpu in 1 2 3; do
    apps=$(nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
    [[ -z "$apps" ]] || { echo "compute process on planned GPU $gpu" >&2; exit 8; }
  done
  [[ "$sample" -eq 3 ]] || sleep 2
done

decode_one() {
  local gpu=$1 name=$2 model=$3
  local out="$ER/generations/$name/output_42.json"
  mkdir -p "$ER/generations/$name"
  CUDA_VISIBLE_DEVICES=$gpu python "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py" \
    --manifest "$MANIFEST" --model "$model" --policy-name "$name" --probe none --output "$out" \
    --seed 42 --temperature 0.7 --top-p 0.9 --max-new-tokens 1024 \
    >"$ER/logs/predecode_${name}.log" 2>&1
}
worker1() { decode_one 1 base "$BASE"; decode_one 1 ronpo_os "$EXP/train/full/ronpo_os/checkpoint-900"; decode_one 1 ronpo_topmass "$EXP/train/full/ronpo_topmass/checkpoint-900"; }
worker2() { decode_one 2 inpo_avg "$EXP/train/full/inpo_avg/checkpoint-900"; decode_one 2 ht_mnpo_harmless "$EXP/train/full/ht_mnpo_harmless/checkpoint-900"; decode_one 2 ht_mnpo_helpful "$EXP/train/stretch/full/ht_mnpo_helpful/checkpoint-900"; }
worker3() { decode_one 3 sppo_avg "$EXP/train/stretch/full/sppo_avg/checkpoint-900"; decode_one 3 simpo "$EXP/train/stretch/full/simpo/checkpoint-900"; decode_one 3 ipo "$EXP/train/stretch/full/ipo/checkpoint-900"; }
worker1 & p1=$!; worker2 & p2=$!; worker3 & p3=$!
wait "$p1" "$p2" "$p3"
date -Is >"$ER/PREDECODE_NINE_COMPLETE"
