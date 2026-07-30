#!/usr/bin/env bash
set -euo pipefail

cd /home/sjkim/MNPO

source /home/sjkim/MNPO/scripts/setup_ext_cache.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH=/home/sjkim/MNPO
export WANDB_PROJECT="${WANDB_PROJECT:-mnpo}"
export WANDB_ENTITY="${WANDB_ENTITY:-promotion-kim}"
export WANDB_MODE="${WANDB_MODE:-online}"
export ACCELERATE_LOG_LEVEL="${ACCELERATE_LOG_LEVEL:-info}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp3s0f0}"

PYTHON_TRAIN="${PYTHON_TRAIN:-/home/sjkim/anaconda3/envs/mnpo_train/bin/python}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-/home/sjkim/MNPO/accelerate_configs/single_gpu.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
TRAIN_PER_DEVICE_BATCH_SIZE="${TRAIN_PER_DEVICE_BATCH_SIZE:-4}"
TRAIN_PER_DEVICE_EVAL_BATCH_SIZE="${TRAIN_PER_DEVICE_EVAL_BATCH_SIZE:-4}"
TRAIN_GRADIENT_ACCUMULATION_STEPS="${TRAIN_GRADIENT_ACCUMULATION_STEPS:-4}"
EVAL_GENERATION_CUDA_VISIBLE_DEVICES="${EVAL_GENERATION_CUDA_VISIBLE_DEVICES:-0}"
BASE_CONFIG="/home/sjkim/MNPO/training_configs/ronpo/qwen2.5-1.5b-instruct-ronpo-multiobj-iter1.yaml"

POLICY_MODEL="${POLICY_MODEL:-/ext_hdd/sjkim/mnpo/outputs_ronpo_h200/qwen2.5-1.5b-instruct_ronpo_vs_htmnpo_online_multiobj_stage_1}"
PREF_DIR="${PREF_DIR:-/ext_hdd/sjkim/mnpo/data/qwen2.5-1.5b-instruct_online_htmnpo_ronpo/ronpo/iter2/precomputed_sigma_relative_pairs2_samples1}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-7}"
OUTPUT_DIR="${OUTPUT_DIR:-/ext_hdd/sjkim/mnpo/outputs_ronpo_h200/qwen2.5-1.5b-instruct_ronpo_stage2_relative_lr1e7_odin2}"
GEN_DIR="${GEN_DIR:-/ext_hdd/sjkim/mnpo/eval_generations/ronpo_stage2_relative_lr1e7_odin2}"
RUN_NAME="${RUN_NAME:-ronpo-s2-lr1e7-od2}"
RUN_SAFE_NAME="$(printf '%s' "$RUN_NAME" | tr -c '[:alnum:]_.-' '_')"
CONFIG="${CONFIG:-/tmp/ronpo_stage2_relative_config_${USER:-sjkim}_${RUN_SAFE_NAME}_$$.yaml}"
LOG_DIR="${LOG_DIR:-/ext_hdd/sjkim/mnpo/logs}"
LOG_PREFIX="${LOG_PREFIX:-ronpo_s2_rel_${RUN_SAFE_NAME}}"
TRAIN_SAVE_TOTAL_LIMIT="${TRAIN_SAVE_TOTAL_LIMIT:-5}"
TRAIN_SAVE_STEPS="${TRAIN_SAVE_STEPS:-100}"
TRAIN_EVAL_STEPS="${TRAIN_EVAL_STEPS:-100}"
TRAIN_LOGGING_STEPS="${TRAIN_LOGGING_STEPS:-5}"
EVAL_GENERATION_SAMPLES="${EVAL_GENERATION_SAMPLES:-20}"
EVAL_GENERATION_MAX_NEW_TOKENS="${EVAL_GENERATION_MAX_NEW_TOKENS:-1024}"
ACCELERATE_MAIN_PROCESS_PORT="${ACCELERATE_MAIN_PROCESS_PORT:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
GENERATE_DURING_EVAL="${GENERATE_DURING_EVAL:-true}"

mkdir -p "$LOG_DIR" "$GEN_DIR"

# Transformers versions differ on the TrainingArguments field name.
sed 's/^evaluation_strategy:/eval_strategy:/' "$BASE_CONFIG" > "$CONFIG"

ACCELERATE_ARGS=(
  -m accelerate.commands.launch
  --config_file "$ACCELERATE_CONFIG" \
  --num_processes="$NUM_PROCESSES" \
)
if [[ -n "$ACCELERATE_MAIN_PROCESS_PORT" ]]; then
  ACCELERATE_ARGS+=(--main_process_port "$ACCELERATE_MAIN_PROCESS_PORT")
fi

RESUME_ARGS=()
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  RESUME_ARGS+=(--resume_from_checkpoint="$RESUME_FROM_CHECKPOINT")
fi

"$PYTHON_TRAIN" "${ACCELERATE_ARGS[@]}" \
  -m mnpo_scripts.run_mnpo \
  "$CONFIG" \
  --model_name_or_path="$POLICY_MODEL" \
  --dataset_mixer="$PREF_DIR:1.0" \
  --output_dir="$OUTPUT_DIR" \
  --run_name="$RUN_NAME" \
  "${RESUME_ARGS[@]}" \
  --loss_type=ronpo \
  --max_history_t=1 \
  --history_weights=1.0 \
  --learning_rate="$LEARNING_RATE" \
  --warmup_ratio=0.1 \
  --per_device_train_batch_size="$TRAIN_PER_DEVICE_BATCH_SIZE" \
  --per_device_eval_batch_size="$TRAIN_PER_DEVICE_EVAL_BATCH_SIZE" \
  --gradient_accumulation_steps="$TRAIN_GRADIENT_ACCUMULATION_STEPS" \
  --max_length=2048 \
  --max_prompt_length=1800 \
  --do_eval=true \
  --eval_steps="$TRAIN_EVAL_STEPS" \
  --save_steps="$TRAIN_SAVE_STEPS" \
  --save_total_limit="$TRAIN_SAVE_TOTAL_LIMIT" \
  --logging_steps="$TRAIN_LOGGING_STEPS" \
  --generate_during_eval="$GENERATE_DURING_EVAL" \
  --eval_generation_samples="$EVAL_GENERATION_SAMPLES" \
  --eval_generation_max_new_tokens="$EVAL_GENERATION_MAX_NEW_TOKENS" \
  --eval_generation_do_sample=false \
  --eval_generation_backend=checkpoint \
  --eval_generation_output_dir="$GEN_DIR" \
  --eval_generation_device=cuda \
  --eval_generation_cuda_visible_devices="$EVAL_GENERATION_CUDA_VISIBLE_DEVICES" \
  --eval_generation_dtype=bfloat16 \
  --eval_generation_keep_snapshot=false \
  --eval_generation_local_files_only=true \
  --eval_generation_print_max_chars=1200 \
  2>&1 | tee "$LOG_DIR/${LOG_PREFIX}_$(date +%Y%m%d_%H%M%S).log"
