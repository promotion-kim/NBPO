#!/usr/bin/env bash
set -euo pipefail

ARM=${ARM:?set ARM}
GPU=${GPU:?set GPU}
SEED=${SEED:-42}
SJ=/NHNHOME/AIPR/sjkim
P=$SJ/MNPO_rev_20260720
W=$SJ/ronpo_gemma_baselines_s1
PY=$SJ/venv_clean/bin/python
BASE=google/gemma-2-2b-it
CACHE=$SJ/baseline_repair_1p5b_20260714/cache/huggingface/hub
OUT=$W/$ARM
mkdir -p "$OUT" "$W/logs"
[[ -f "$OUT/all_results.json" ]] && exit 0

case "$ARM" in
  inpo|sppo|mnpo|dpo|ipo|simpo) ;;
  *) echo "unsupported arm: $ARM" >&2; exit 2 ;;
esac

export PYTHONPATH=$P
export HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface
export HF_HUB_CACHE=$CACHE HUGGINGFACE_HUB_CACHE=$CACHE
export WANDB_ENTITY=promotion-kim WANDB_PROJECT=mnpo WANDB_MODE=online WANDB_DIR=$W
export TOKENIZERS_PARALLELISM=false MNPO_DISABLE_APEX=1 NCCL_SOCKET_IFNAME=lo

resume=()
last=$(find "$OUT" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null | sort -V | tail -1)
[[ -n "$last" ]] && resume=(--resume_from_checkpoint="$OUT/$last")

CUDA_VISIBLE_DEVICES=$GPU "$PY" -m accelerate.commands.launch \
  --config_file "$P/accelerate_configs/single_gpu.yaml" --num_processes=1 \
  --main_process_port=$((30400 + GPU + SEED % 10)) -m mnpo_scripts.run_mnpo \
  "$P/training_configs/ronpo/qwen2.5-1.5b-instruct-ronpo-multiobj-iter1.yaml" \
  --model_name_or_path="$BASE" --dataset_mixer="$W/precomputed:1.0" \
  --loss_type="$ARM" --max_history_t=1 --history_weights=1.0 \
  --learning_rate=5.0e-7 --warmup_ratio=0.1 --num_train_epochs=1 --seed="$SEED" \
  --max_steps=1800 --per_device_train_batch_size=4 --gradient_accumulation_steps=4 \
  --gradient_checkpointing=true --max_length=2048 --max_prompt_length=1800 \
  --do_eval=false --eval_strategy=no --generate_during_eval=false \
  --save_strategy=steps --save_steps=300 --save_total_limit=2 --logging_steps=10 \
  --output_dir="$OUT" --run_name="${ARM}-gemma-2b-uf5-stage1-s${SEED}" \
  "${resume[@]}" > "$W/logs/train_${ARM}_s${SEED}.log" 2>&1

test -f "$OUT/all_results.json"
