#!/usr/bin/env bash
set -euo pipefail

GPUS=${GPUS:-0,1,2,3}
NPROC=${NPROC:-4}
SJ=/NHNHOME/AIPR/sjkim
P=$SJ/MNPO_rev_20260720
SRC=$SJ/ronpo_gemma_20260720/pairs
W=$SJ/ronpo_gemma_baselines_s1
PY=$SJ/venv_clean/bin/python
BASE=google/gemma-2-2b-it
CACHE=$SJ/baseline_repair_1p5b_20260714/cache/huggingface/hub
mkdir -p "$W/logs"
[[ -f "$W/precomputed/dataset_dict.json" ]] && exit 0

export PYTHONPATH=$P
export HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface
export HF_HUB_CACHE=$CACHE HUGGINGFACE_HUB_CACHE=$CACHE
export TOKENIZERS_PARALLELISM=false MNPO_DISABLE_APEX=1 NCCL_SOCKET_IFNAME=lo

CUDA_VISIBLE_DEVICES=$GPUS "$PY" -m accelerate.commands.launch --num_processes="$NPROC" \
  --main_process_port=30310 -m mnpo_scripts.precompute \
  --model_name_or_path "$BASE" --ref_model "$BASE" --history_paths "$BASE" \
  --train_dir "$SRC/train_mnpo_unused.jsonl" --test_dir "$SRC/test_mnpo_unused.jsonl" \
  --output_dir "$W/precomputed" --per_device_train_batch_size 8 \
  --max_length 2048 --max_prompt_length 1800 --apply_chat_template true \
  --auto_insert_empty_system_msg false --ronpo_target_mode none --sanity_check False \
  > "$W/logs/precompute.log" 2>&1

test -f "$W/precomputed/dataset_dict.json"
date -Is > "$W/precomputed/READY"
