#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 2 ]]; then
  echo "usage: $0 full_atom|k_only GPU_ID [seed]" >&2
  exit 2
fi
MODE="$1"
GPU_ID="$2"
SEED="${3:-42}"

case "$MODE" in
  full_atom)
    RUN_NAME="ronpo-conflict-full-s1"
    ;;
  k_only)
    RUN_NAME="ronpo-conflict-konly-s1"
    ;;
  *)
    echo "Unsupported MODE=$MODE. Expected full_atom or k_only." >&2
    exit 2
    ;;
esac

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

EXP_ROOT="${EXP_ROOT:-$PROJECT_ROOT/experiments/ronpo_conflict_tier0_20260629}"
PAIR_ROOT="$EXP_ROOT/pairs/$MODE"
OUT_ROOT="${OUT_ROOT:-/ext_hdd/sjkim/mnpo/experiments/ronpo_conflict_tier0_20260629}"
PREF_DIR="$OUT_ROOT/precomputed_${MODE}_seed${SEED}"
OUTPUT_DIR="$OUT_ROOT/outputs/${RUN_NAME}_seed${SEED}"
LOG_DIR="$OUT_ROOT/logs"
LOG_FILE="$LOG_DIR/${RUN_NAME}_gpu${GPU_ID}_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUT_ROOT" "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[start] $(date -Is) mode=$MODE gpu=$GPU_ID seed=$SEED"
echo "[paths] pairs=$PAIR_ROOT"
echo "[paths] precomputed=$PREF_DIR"
echo "[paths] output=$OUTPUT_DIR"
echo "[paths] log=$LOG_FILE"
nvidia-smi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] setup complete; exiting before precompute/train."
  exit 0
fi

if [[ ! -f "$PAIR_ROOT/train_ronpo.jsonl" || ! -f "$PAIR_ROOT/test_ronpo.jsonl" ]]; then
  echo "Missing pair files in $PAIR_ROOT" >&2
  exit 1
fi

if [[ ! -d "$PREF_DIR" ]]; then
  echo "[precompute] $(date -Is)"
  "$PYTHON_TRAIN" -m accelerate.commands.launch \
    --config_file "$ACCELERATE_CONFIG" \
    --num_processes=1 \
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
    --ronpo_target_mode none \
    --sanity_check False
else
  echo "[precompute] reuse existing $PREF_DIR"
fi

echo "[train] $(date -Is)"
ACCELERATE_LOG_LEVEL=info "$PYTHON_TRAIN" -m accelerate.commands.launch \
  --config_file "$ACCELERATE_CONFIG" \
  --num_processes=1 \
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
  --generate_during_eval=true \
  --eval_generation_samples="${EVAL_GENERATION_SAMPLES:-5}" \
  --eval_generation_max_new_tokens="${EVAL_GENERATION_MAX_NEW_TOKENS:-256}" \
  --eval_generation_do_sample=false \
  --eval_generation_backend=checkpoint \
  --eval_generation_output_dir="$OUT_ROOT/eval_generations" \
  --eval_generation_cuda_visible_devices="$GPU_ID" \
  --eval_generation_device=cuda \
  --eval_generation_dtype=bfloat16 \
  --eval_generation_keep_snapshot=false \
  --eval_generation_local_files_only=true \
  --eval_generation_print_max_chars=1200

echo "[done] $(date -Is) mode=$MODE"
