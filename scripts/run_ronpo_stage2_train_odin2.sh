#!/usr/bin/env bash
set -euo pipefail

cd /home/sjkim/MNPO

source /home/sjkim/MNPO/scripts/setup_ext_cache.sh

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=/home/sjkim/MNPO
export WANDB_PROJECT=mnpo
export WANDB_ENTITY=promotion-kim
export WANDB_MODE=online
export ACCELERATE_LOG_LEVEL=info
export NCCL_P2P_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=enp3s0f0

PYTHON_TRAIN=/home/sjkim/anaconda3/envs/mnpo_train/bin/python
ACCELERATE_CONFIG=/home/sjkim/MNPO/accelerate_configs/deepspeed_zero3.yaml
CONFIG=/home/sjkim/MNPO/training_configs/ronpo/qwen2.5-1.5b-instruct-ronpo-multiobj-iter1.yaml
POLICY_MODEL=/ext_hdd/sjkim/mnpo/outputs_ronpo_h200/qwen2.5-1.5b-instruct_ronpo_vs_htmnpo_online_multiobj_stage_1
PREF_DIR=/ext_hdd/sjkim/mnpo/data/qwen2.5-1.5b-instruct_online_htmnpo_ronpo/ronpo/iter2/precomputed_sigma_best_vs_adversary_pairs1_samples0
OUTPUT_DIR=/ext_hdd/sjkim/mnpo/outputs_ronpo_h200/qwen2.5-1.5b-instruct_ronpo_vs_htmnpo_online_multiobj_stage_2
RUN_NAME=qwen2.5-1.5b-instruct_ronpo_vs_htmnpo_online_multiobj_stage_2_odin2_zero3
LOG_DIR=/ext_hdd/sjkim/mnpo/logs

mkdir -p "$LOG_DIR"

"$PYTHON_TRAIN" -m accelerate.commands.launch \
  --config_file "$ACCELERATE_CONFIG" \
  --num_processes=2 \
  -m mnpo_scripts.run_mnpo \
  "$CONFIG" \
  --model_name_or_path="$POLICY_MODEL" \
  --dataset_mixer="$PREF_DIR:1.0" \
  --output_dir="$OUTPUT_DIR" \
  --run_name="$RUN_NAME" \
  --max_history_t=1 \
  --history_weights=1.0 \
  --per_device_train_batch_size=8 \
  --per_device_eval_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --max_length=2048 \
  --max_prompt_length=1800 \
  --generate_during_eval=false \
  --eval_generation_samples=5 \
  --eval_generation_max_new_tokens=256 \
  --eval_generation_do_sample=false \
  --eval_generation_temperature=0.7 \
  --eval_generation_top_p=0.9 \
  --eval_generation_top_k=20 \
  --eval_generation_backend=checkpoint \
  --eval_generation_output_dir=/ext_hdd/sjkim/mnpo/eval_generations \
  --eval_generation_device=cuda \
  --eval_generation_dtype=bfloat16 \
  --eval_generation_keep_snapshot=false \
  --eval_generation_local_files_only=true \
  --eval_generation_print_max_chars=1200 \
  --eval_steps=100 \
  --save_steps=100 \
  --save_total_limit=5 \
  --logging_steps=5 \
  2>&1 | tee "$LOG_DIR/ronpo_stage2_odin2_gpu01_$(date +%Y%m%d_%H%M%S).log"
