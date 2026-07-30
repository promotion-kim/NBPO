#!/usr/bin/env bash
# OS-RONPO vs top-mass controlled A/B: identical base init + data + logps,
# only --ronpo_target_column differs. One single-GPU accelerate run per arm.
#
# Usage:
#   GPU=1 TARGET_COL=target_topmass_k0p05 OUT=/ext_hdd/.../topmass RUN=os-topmass \
#     [MAX_STEPS=6] bash scripts/run_os_ronpo_arm.sh
set -euo pipefail
cd /home/sjkim/MNPO
source /home/sjkim/MNPO/scripts/setup_ext_cache.sh

GPU="${GPU:?set GPU}"
TARGET_COL="${TARGET_COL:?set TARGET_COL}"
OUT="${OUT:?set OUT}"
RUN="${RUN:?set RUN}"
MAX_STEPS="${MAX_STEPS:-0}"
PDBS="${PDBS:-4}"
GA="${GA:-4}"
GRADCKPT="${GRADCKPT:-true}"
PORT="${PORT:-$((29600 + GPU))}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH=/home/sjkim/MNPO
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=mnpo
export WANDB_ENTITY=promotion-kim
export WANDB_MODE="${WANDB_MODE:-online}"

PYTHON="${PYTHON:-/home/sjkim/anaconda3/envs/mnpo_train/bin/python}"
DATASET="${DATASET:-/ext_hdd/sjkim/mnpo/data/os_ronpo_iter2_targets}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
LOGDIR="${LOGDIR:-/ext_hdd/sjkim/mnpo/logs}"
mkdir -p "$LOGDIR" "$OUT"

EXTRA=()
if [[ "$MAX_STEPS" != "0" ]]; then
  EXTRA+=(--max_steps="$MAX_STEPS" --save_strategy=no --report_to=none)
fi

"$PYTHON" -m accelerate.commands.launch \
  --config_file accelerate_configs/single_gpu.yaml \
  --num_processes=1 --main_process_port="$PORT" \
  -m mnpo_scripts.run_mnpo \
  training_configs/ronpo/qwen2.5-1.5b-instruct-ronpo-multiobj-iter1.yaml \
  --model_name_or_path="$MODEL" \
  --dataset_mixer="$DATASET:1.0" \
  --loss_type=ronpo \
  --ronpo_alpha=1.0 --ronpo_tau=0.05 \
  --ronpo_target_column="$TARGET_COL" \
  --max_history_t=1 --history_weights=1.0 \
  --learning_rate=5.0e-7 --warmup_ratio=0.1 --num_train_epochs=1 \
  --per_device_train_batch_size="$PDBS" --gradient_accumulation_steps="$GA" \
  --gradient_checkpointing="$GRADCKPT" \
  --max_length=2048 --max_prompt_length=1800 \
  --do_eval=false --eval_strategy=no --generate_during_eval=false \
  --save_strategy=steps --save_steps=800 --save_total_limit=2 \
  --logging_steps=10 \
  --output_dir="$OUT" --run_name="$RUN" \
  "${EXTRA[@]}" \
  2>&1 | tee "$LOGDIR/${RUN}_$(date +%Y%m%d_%H%M%S).log"
