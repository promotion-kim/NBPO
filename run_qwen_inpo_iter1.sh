#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${USE_EXT_CACHE:-1}" == "1" ]]; then
  # shellcheck source=scripts/setup_ext_cache.sh
  source "$PROJECT_ROOT/scripts/setup_ext_cache.sh"
fi
PYTHON_TRAIN="${PYTHON_TRAIN:-/home/sjkim/anaconda3/envs/mnpo_train/bin/python}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-$PROJECT_ROOT/data/gemma2_ufb_part1_train.jsonl}"
TEST_FILE="${TEST_FILE:-$PROJECT_ROOT/data/gemma2_ufb_part1_test.jsonl}"
PREF_DIR="${PREF_DIR:-$PROJECT_ROOT/data/qwen2.5-1.5b-instruct_iter1_precomputed}"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/training_configs/inpo/qwen2.5-1.5b-instruct-inpo-iter1.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$PROJECT_ROOT/accelerate_configs/deepspeed_zero3.yaml}"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp3s0f0}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/ext_hdd/sjkim/mnpo/triton_cache}"
export WANDB_ENTITY="${WANDB_ENTITY:-promotion-kim}"
export WANDB_PROJECT="${WANDB_PROJECT:-mnpo}"

"$PYTHON_TRAIN" -m accelerate.commands.launch --num_processes=2 -m mnpo_scripts.precompute \
  --model_name_or_path "$BASE_MODEL" \
  --ref_model "$BASE_MODEL" \
  --history_paths "$BASE_MODEL" \
  --train_dir "$TRAIN_FILE" \
  --test_dir "$TEST_FILE" \
  --output_dir "$PREF_DIR" \
  --per_device_train_batch_size 2 \
  --max_length 2048 \
  --max_prompt_length 1800 \
  --sanity_check False

ACCELERATE_LOG_LEVEL=info "$PYTHON_TRAIN" -m accelerate.commands.launch \
  --config_file "$ACCELERATE_CONFIG" \
  -m mnpo_scripts.run_mnpo \
  "$CONFIG_FILE"
