#!/usr/bin/env bash
set -euo pipefail

name="$1"
model="$2"
port="$3"
root=/NHNHOME/AIPR/sjkim/novelty_defense_20260723
repo=/NHNHOME/AIPR/sjkim/MNPO_rev_20260720
work="$root/ifeval/$name"
mkdir -p "$work" "$root/logs/ifeval/$name"
cd "$work"

USE_EXT_CACHE=0 \
MODEL_NAME="$model" \
MODEL_BASENAME="$name" \
GPU_IDS="${CUDA_VISIBLE_DEVICES:?}" \
PORT="$port" \
TENSOR_PARALLEL_SIZE=1 \
LOG_DIR="$root/logs/ifeval/$name" \
EVAL_PYTHON="$root/eval_venv/bin/python" \
VLLM_PYTHON=/NHNHOME/AIPR/sjkim/venv_clean/bin/python \
EVAL_PYTHONPATH="$root/eval_venv/lib/python3.12/site-packages" \
VLLM_PYTHONPATH="$repo" \
MAX_MODEL_LEN=4096 \
EXTRA_VLLM_ARGS="--gpu-memory-utilization 0.9 --dtype bfloat16" \
TASK_ARGS="--datasets ifeval --eval-batch-size 20 --generation-seed 42" \
TASK_SCRIPT="$root/code/run_rule_based_task.py" \
bash "$root/code/run_vllm_eval.sh"
