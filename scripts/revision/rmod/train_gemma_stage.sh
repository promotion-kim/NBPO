#!/usr/bin/env bash
set -euo pipefail

GPU=${GPU:?set GPU}
GPUS=${GPUS:-$GPU}
NPROC=${NPROC:-1}
STAGE=${STAGE:?set STAGE}
INIT=${INIT:?set INIT checkpoint}
DATASET=${DATASET:?set DATASET}
OUT=${OUT:?set OUT}
SEED=${SEED:-42}
TARGET_COLUMN=${TARGET_COLUMN:-ronpo_target}
MAX_STEPS=${MAX_STEPS:-1800}

SJ=/NHNHOME/AIPR/sjkim
P=$SJ/MNPO_rev_20260720
PY=$SJ/venv_clean/bin/python
CACHE=$SJ/baseline_repair_1p5b_20260714/cache/huggingface/hub
mkdir -p "$OUT" "$(dirname "$OUT")/logs"
[[ -f "$OUT/all_results.json" ]] && exit 0

export PYTHONPATH=$P
export HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface
export HF_HUB_CACHE=$CACHE HUGGINGFACE_HUB_CACHE=$CACHE
export WANDB_ENTITY=promotion-kim WANDB_PROJECT=mnpo WANDB_MODE=online WANDB_DIR=$(dirname "$OUT")
export TOKENIZERS_PARALLELISM=false MNPO_DISABLE_APEX=1 NCCL_SOCKET_IFNAME=lo

resume=()
last=$(find "$OUT" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null | sort -V | tail -1)
[[ -n "$last" ]] && resume=(--resume_from_checkpoint="$OUT/$last")
log="$OUT/train.log"

if (( NPROC > 1 )); then
  ACCEL_CONFIG=$P/accelerate_configs/multi_gpu.yaml
  PER_DEVICE_BATCH=1
  GRAD_ACC=4
else
  ACCEL_CONFIG=$P/accelerate_configs/single_gpu.yaml
  PER_DEVICE_BATCH=4
  GRAD_ACC=4
fi

CUDA_VISIBLE_DEVICES=$GPUS "$PY" -m accelerate.commands.launch \
  --config_file "$ACCEL_CONFIG" --num_processes="$NPROC" \
  --main_process_port=$((30100 + GPU + 10 * STAGE + SEED % 10)) \
  -m mnpo_scripts.run_mnpo \
  "$P/training_configs/ronpo/qwen2.5-1.5b-instruct-ronpo-multiobj-iter1.yaml" \
  --model_name_or_path="$INIT" --dataset_mixer="$DATASET:1.0" \
  --loss_type=ronpo --ronpo_alpha=1.0 --ronpo_tau=0.05 --ronpo_target_column="$TARGET_COLUMN" \
  --max_history_t=1 --history_weights=1.0 --learning_rate=5.0e-7 --warmup_ratio=0.1 \
  --num_train_epochs=1 --seed="$SEED" --max_steps="$MAX_STEPS" \
  --per_device_train_batch_size="$PER_DEVICE_BATCH" --gradient_accumulation_steps="$GRAD_ACC" --gradient_checkpointing=true \
  --max_length=2048 --max_prompt_length=1800 --do_eval=false --eval_strategy=no \
  --generate_during_eval=false --save_strategy=steps --save_steps=300 --save_total_limit=2 \
  --logging_steps=10 --output_dir="$OUT" --run_name="ronpo-gemma-2b-os-stage${STAGE}-s${SEED}-${NPROC}gpu-${MAX_STEPS}steps" \
  "${resume[@]}" > "$log" 2>&1

test -f "$OUT/all_results.json"
