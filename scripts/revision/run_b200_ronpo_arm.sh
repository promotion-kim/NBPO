#!/usr/bin/env bash
# Single-GPU stage-2 RONPO arm on a B200 host. Matched A/B against the shipped
# arms: same dataset, pairs, and logps; only the target column and/or training
# seed differ.
#   GPU=0 TARGET_COL=target_os_k0p05_lam4 NAME=os-ronpo-os-k05-lam4 [SEED=42] \
#     bash run_b200_ronpo_arm.sh
set -euo pipefail
GPU="${GPU:?}"; TARGET_COL="${TARGET_COL:?}"; NAME="${NAME:?}"
SEED="${SEED:-42}"
SJ=/NHNHOME/AIPR/sjkim
PROJECT_ROOT="$SJ/MNPO_rev_20260720"
DATASET="$SJ/data_1p5b/os_ronpo_iter2_targets_lam"
OUT="$SJ/ronpo_arms_20260720/$NAME"
LOGDIR="$SJ/ronpo_arms_20260720/logs"
mkdir -p "$OUT" "$LOGDIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$PROJECT_ROOT"
export HF_HOME="$SJ/baseline_repair_1p5b_20260714/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export WANDB_ENTITY=promotion-kim WANDB_PROJECT=mnpo WANDB_MODE=online
export WANDB_DIR="$SJ/ronpo_arms_20260720"
export TOKENIZERS_PARALLELISM=false MNPO_DISABLE_APEX=1 NCCL_SOCKET_IFNAME=lo
PY="$SJ/venv_clean/bin/python"

"$PY" -m accelerate.commands.launch \
  --config_file "$PROJECT_ROOT/accelerate_configs/single_gpu.yaml" \
  --num_processes=1 --main_process_port="$((29800 + GPU))" \
  -m mnpo_scripts.run_mnpo \
  "$PROJECT_ROOT/training_configs/ronpo/qwen2.5-1.5b-instruct-ronpo-multiobj-iter1.yaml" \
  --model_name_or_path=Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_mixer="$DATASET:1.0" \
  --loss_type=ronpo --ronpo_alpha=1.0 --ronpo_tau=0.05 \
  --ronpo_target_column="$TARGET_COL" \
  --max_history_t=1 --history_weights=1.0 \
  --learning_rate=5.0e-7 --warmup_ratio=0.1 --num_train_epochs=1 \
  --per_device_train_batch_size=4 --gradient_accumulation_steps=4 \
  --gradient_checkpointing=true \
  --max_length=2048 --max_prompt_length=1800 \
  --seed="$SEED" \
  --do_eval=false --eval_strategy=no --generate_during_eval=false \
  --save_strategy=steps --save_steps=800 --save_total_limit=2 \
  --logging_steps=10 \
  --output_dir="$OUT" --run_name="$NAME" \
  2>&1 | tee "$LOGDIR/${NAME}_$(date +%m%d_%H%M).log"
