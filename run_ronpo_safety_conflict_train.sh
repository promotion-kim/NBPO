#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 2 ]]; then
  echo "usage: $0 full_atom|k_only|uniform|a_only|maxmin_pointwise|full_sample|full_all|full_expect|k_sample|k_all|k_expect GPU_ID [seed]" >&2
  exit 2
fi

MODE="$1"
GPU_ID="$2"
SEED="${3:-42}"
case "$MODE" in
  full_atom)
    RUN_NAME="ronpo-safe-full-s1"
    ;;
  k_only)
    RUN_NAME="ronpo-safe-konly-s1"
    ;;
  uniform)
    RUN_NAME="ronpo-safe-uniform-s1"
    ;;
  a_only)
    RUN_NAME="ronpo-safe-aonly-s1"
    ;;
  maxmin_pointwise)
    RUN_NAME="ronpo-safe-maxmin-s1"
    ;;
  k_sample)
    RUN_NAME="ronpo-safe-ksample-s1"
    ;;
  k_all)
    RUN_NAME="ronpo-safe-kall-s1"
    ;;
  k_expect)
    RUN_NAME="ronpo-safe-kexpect-s1"
    ;;
  full_sample)
    RUN_NAME="ronpo-safe-sample-s1"
    ;;
  full_all)
    RUN_NAME="ronpo-safe-all-s1"
    ;;
  full_expect)
    RUN_NAME="ronpo-safe-expect-s1"
    ;;
  *)
    echo "Unsupported MODE=$MODE. Expected full_atom, k_only, uniform, a_only, maxmin_pointwise, full_sample, full_all, full_expect, k_sample, k_all, or k_expect." >&2
    exit 2
    ;;
esac
RUN_NAME="${RUN_NAME_OVERRIDE:-$RUN_NAME}"

if [[ -f "$PROJECT_ROOT/scripts/setup_ext_cache.sh" ]]; then
  # shellcheck source=scripts/setup_ext_cache.sh
  source "$PROJECT_ROOT/scripts/setup_ext_cache.sh"
fi

export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-promotion-kim}"
export WANDB_PROJECT="${WANDB_PROJECT:-mnpo}"

PYTHON_TRAIN="${PYTHON_TRAIN:-/home/sjkim/anaconda3/envs/mnpo_train/bin/python}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
CONFIG="${CONFIG:-$PROJECT_ROOT/training_configs/ronpo/qwen2.5-1.5b-instruct-ronpo-multiobj-iter1.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$PROJECT_ROOT/accelerate_configs/single_gpu.yaml}"
PRECOMPUTE_ACCELERATE_CONFIG="${PRECOMPUTE_ACCELERATE_CONFIG:-$ACCELERATE_CONFIG}"
TRAIN_ACCELERATE_CONFIG="${TRAIN_ACCELERATE_CONFIG:-$ACCELERATE_CONFIG}"
PRECOMPUTE_NUM_PROCESSES="${PRECOMPUTE_NUM_PROCESSES:-1}"
TRAIN_NUM_PROCESSES="${TRAIN_NUM_PROCESSES:-1}"

EXP_ROOT="${EXP_ROOT:-$PROJECT_ROOT/experiments/ronpo_safety_conflict_qwen25_1p5b_20260629}"
PAIR_ROOT="$EXP_ROOT/pairs/$MODE"
OUT_ROOT="${OUT_ROOT:-/ext_hdd/sjkim/mnpo/experiments/ronpo_safety_conflict_qwen25_1p5b_20260629}"
PREF_DIR="$OUT_ROOT/precomputed_${MODE}_seed${SEED}"
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-$OUT_ROOT/outputs/${RUN_NAME}_seed${SEED}}"
LOG_DIR="$OUT_ROOT/logs"
LOG_FILE="$LOG_DIR/${RUN_NAME}_gpu${GPU_ID}_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  RESUME_ARGS+=(--resume_from_checkpoint="$RESUME_FROM_CHECKPOINT")
fi
TRAIN_EXTRA_ARGS=()
if [[ -n "${MAX_STEPS:-}" ]]; then
  TRAIN_EXTRA_ARGS+=(--max_steps="$MAX_STEPS")
fi
TRAIN_LAUNCH_ARGS=()
if [[ -n "${ACCELERATE_MAIN_PROCESS_PORT:-}" ]]; then
  TRAIN_LAUNCH_ARGS+=(--main_process_port "$ACCELERATE_MAIN_PROCESS_PORT")
fi
PRECOMPUTE_LAUNCH_ARGS=()
if [[ -n "${ACCELERATE_MAIN_PROCESS_PORT:-}" ]]; then
  PRECOMPUTE_LAUNCH_ARGS+=(--main_process_port "$ACCELERATE_MAIN_PROCESS_PORT")
fi

mkdir -p "$OUT_ROOT" "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[start] $(date -Is) mode=$MODE gpu=$GPU_ID seed=$SEED"
echo "[paths] pairs=$PAIR_ROOT"
echo "[paths] precomputed=$PREF_DIR"
echo "[paths] output=$OUTPUT_DIR"
echo "[paths] log=$LOG_FILE"
if [[ ${#RESUME_ARGS[@]} -gt 0 ]]; then
  echo "[resume] $RESUME_FROM_CHECKPOINT"
fi
nvidia-smi

if [[ ! -f "$PAIR_ROOT/train_ronpo.jsonl" || ! -f "$PAIR_ROOT/test_ronpo.jsonl" ]]; then
  echo "Missing pair files in $PAIR_ROOT" >&2
  exit 1
fi

if [[ ! -d "$PREF_DIR" ]]; then
  echo "[precompute] $(date -Is)"
  "$PYTHON_TRAIN" -m accelerate.commands.launch \
    --config_file "$PRECOMPUTE_ACCELERATE_CONFIG" \
    --num_processes="$PRECOMPUTE_NUM_PROCESSES" \
    "${PRECOMPUTE_LAUNCH_ARGS[@]}" \
    -m mnpo_scripts.precompute \
    --model_name_or_path "$BASE_MODEL" \
    --ref_model "$BASE_MODEL" \
    --history_paths "$BASE_MODEL" \
    --train_dir "$PAIR_ROOT/train_ronpo.jsonl" \
    --test_dir "$PAIR_ROOT/test_ronpo.jsonl" \
    --output_dir "$PREF_DIR" \
    --per_device_train_batch_size "${PRECOMPUTE_BATCH_SIZE:-4}" \
    --per_device_eval_batch_size "${PRECOMPUTE_BATCH_SIZE:-4}" \
    --max_length "${MAX_LENGTH:-2048}" \
    --max_prompt_length "${MAX_PROMPT_LENGTH:-1800}" \
    --apply_chat_template true \
    --auto_insert_empty_system_msg false \
    --ronpo_target_mode=none \
    --sanity_check=False
else
  echo "[precompute] reuse existing $PREF_DIR"
fi

echo "[train] $(date -Is)"
ACCELERATE_LOG_LEVEL=info "$PYTHON_TRAIN" -m accelerate.commands.launch \
  --config_file "$TRAIN_ACCELERATE_CONFIG" \
  --num_processes="$TRAIN_NUM_PROCESSES" \
  "${TRAIN_LAUNCH_ARGS[@]}" \
  -m mnpo_scripts.run_mnpo \
  "$CONFIG" \
  --model_name_or_path="$BASE_MODEL" \
  --dataset_mixer="$PREF_DIR:1.0" \
  --output_dir="$OUTPUT_DIR" \
  --run_name="$RUN_NAME" \
  --seed="$SEED" \
  --max_history_t=1 \
  --history_weights=1.0 \
  --loss_type=ronpo \
  --ronpo_alpha=1.0 \
  --ronpo_tau=0.05 \
  --ronpo_target_column=ronpo_target \
  --per_device_train_batch_size="${TRAIN_BATCH_SIZE:-4}" \
  --per_device_eval_batch_size="${TRAIN_EVAL_BATCH_SIZE:-4}" \
  --gradient_accumulation_steps="${GRAD_ACCUM:-4}" \
  --max_length="${MAX_LENGTH:-2048}" \
  --max_prompt_length="${MAX_PROMPT_LENGTH:-1800}" \
  --learning_rate="${LEARNING_RATE:-5.0e-7}" \
  --warmup_ratio="${WARMUP_RATIO:-0.1}" \
  --evaluation_strategy=steps \
  --eval_steps="${EVAL_STEPS:-100}" \
  --save_steps="${SAVE_STEPS:-100}" \
  --save_total_limit="${SAVE_TOTAL_LIMIT:-3}" \
  --logging_steps="${LOGGING_STEPS:-5}" \
  "${RESUME_ARGS[@]}" \
  "${TRAIN_EXTRA_ARGS[@]}" \
  --generate_during_eval="${GENERATE_DURING_EVAL:-true}" \
  --eval_generation_samples="${EVAL_GENERATION_SAMPLES:-5}" \
  --eval_generation_max_new_tokens="${EVAL_GENERATION_MAX_NEW_TOKENS:-256}" \
  --eval_generation_do_sample="${EVAL_GENERATION_DO_SAMPLE:-false}" \
  --eval_generation_backend="${EVAL_GENERATION_BACKEND:-checkpoint}" \
  --eval_generation_output_dir="$OUT_ROOT/eval_generations" \
  --eval_generation_cuda_visible_devices="${EVAL_GENERATION_CUDA_VISIBLE_DEVICES:-$GPU_ID}" \
  --eval_generation_device="${EVAL_GENERATION_DEVICE:-cuda}" \
  --eval_generation_dtype="${EVAL_GENERATION_DTYPE:-bfloat16}" \
  --eval_generation_keep_snapshot="${EVAL_GENERATION_KEEP_SNAPSHOT:-false}" \
  --eval_generation_local_files_only="${EVAL_GENERATION_LOCAL_FILES_ONLY:-true}" \
  --eval_generation_print_max_chars="${EVAL_GENERATION_PRINT_MAX_CHARS:-1200}"

echo "[done] $(date -Is) mode=$MODE"
