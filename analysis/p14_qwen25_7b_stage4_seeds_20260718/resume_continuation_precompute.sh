#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 8 ]] || { echo "usage: $0 PROJECT TRAIN_PY ROOT CACHE SEED STAGE ARM GPU" >&2; exit 2; }
PROJECT=$1; PY=$2; ROOT=$3; CACHE=$4; SEED=$5; STAGE=$6; ARM=$7; GPU=$8
BASE=$CACHE/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
PARENT=$ROOT/seeds/s$SEED/stage$((STAGE-1))/$ARM/train/full
POOL=$ROOT/seeds/s$SEED/stage$STAGE/$ARM/pool
[[ -s $POOL/pairs_train.jsonl && -s $POOL/pairs_test.jsonl && -f $PARENT/config.json ]] || exit 1
export PYTHONPATH=$PROJECT HF_HOME=$CACHE TOKENIZERS_PARALLELISM=false
export TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1 MNPO_DISABLE_APEX=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
test ! -e "$POOL/precompute/logps_resume" || { echo "resume output already exists" >&2; exit 1; }
test ! -e "$POOL/precompute/targets_resume" || { echo "resume target output already exists" >&2; exit 1; }
CUDA_VISIBLE_DEVICES=$GPU "$PY" -m accelerate.commands.launch --config_file "$PROJECT/accelerate_configs/single_gpu.yaml" \
  --num_processes=1 -m mnpo_scripts.precompute --model_name_or_path "$BASE" --ref_model "$BASE" \
  --history_paths "$PARENT" --train_dir "$POOL/pairs_train.jsonl" --test_dir "$POOL/pairs_test.jsonl" \
  --output_dir "$POOL/precompute/logps_resume" --per_device_train_batch_size 2 --per_device_eval_batch_size 2 \
  --max_length 2048 --max_prompt_length 1024 --apply_chat_template true \
  --auto_insert_empty_system_msg false --ronpo_target_mode none --report_to none \
  > "$POOL/logs/precompute_resume.log" 2>&1
KAPPA=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["confirmatory_os_kappa"])' "$ROOT/kappa_lock.json")
"$PY" "$PROJECT/mnpo_scripts/build_os_ronpo_targets.py" --input_dir "$POOL/precompute/logps_resume" \
  --output_dir "$POOL/precompute/targets_resume" --kappas "$KAPPA" --num_proc 12 > "$POOL/logs/build_targets_resume.log" 2>&1
rmdir "$POOL/precompute/logps" 2>/dev/null || true
rmdir "$POOL/precompute/targets" 2>/dev/null || true
ln -sfn "$POOL/precompute/logps_resume" "$POOL/precompute/logps"
ln -sfn "$POOL/precompute/targets_resume" "$POOL/precompute/targets"
date -Is > "$POOL/PREPARED"
