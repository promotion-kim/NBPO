#!/usr/bin/env bash
# Anchored BPO stage driver -- FIXED-REFERENCE (beta = infinity) LEGACY BASELINE.
# NOT the finite-temperature NBPO Algorithm 1 (that stack is scripts/nbpo/,
# driven by scripts/nbpo/run_nbpo_stage.py): weights here are static batch
# rules on fixed-anchor surpluses, training reuses loss_type=ht_mnpo with
# auxiliary anchor/pref-SFT losses, and there is no dual descent or stage gate.
# Usage: run_bpo_stage.sh PHASE ROOT STAGE [ARGS...]
#   decode     ROOT STAGE PARENT GPU
#   judge      ROOT STAGE POLICY_DIR GPU        (POLICY_DIR=- uses P4 base seeds 42/43)
#   build      ROOT STAGE POLICY_DIR
#   precompute ROOT STAGE PARENT GPU
#   train      ROOT STAGE ARM PARENT GPU        (ARM in nbs|ks|unif; smoke then full)
set -euo pipefail
PHASE=$1; ROOT=$2; STAGE=$3; shift 3
P=${P:-/NHNHOME/AIPR/sjkim/MNPO_rev_20260710}
V=${V:-/NHNHOME/AIPR/sjkim/venv_clean}
BASE=${BASE:-/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1/base_objective_screen/hf_ipv4/llama31}
P4=$P/results/p4_8b_saferlhf_table4_20260717
JUDGE=${JUDGE:-/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1/flagship_20260712/cache/huggingface/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a}
SD=$ROOT/stage$STAGE
export PYTHONPATH=$P HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1
mkdir -p "$SD"

case "$PHASE" in
decode)
  PARENT=$1; GPU=$2
  for s in 42 43; do
    out=$SD/gens/seed$s/output_$s.json; mkdir -p "$(dirname "$out")"
    [ -s "$out" ] && continue
    CUDA_VISIBLE_DEVICES=$GPU "$V/bin/python" "$P/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py" \
      --manifest "$P4/dataset_manifest/train_conflict.jsonl" --model "$PARENT" \
      --policy-name "bpo_stage${STAGE}_seed${s}" --probe none --output "$out" --seed "$s" \
      --temperature 0.7 --top-p 0.9 --max-new-tokens 512 --max-model-len 8192 \
      --gpu-memory-utilization 0.88 > "$SD/gens/decode_$s.log" 2>&1
  done
  ;;
judge)
  PD=$1; GPU=$2
  if [ "$PD" = "-" ]; then
    PF="42=$P4/train_pool/generations/seed42/output_42.json 43=$P4/train_pool/generations/seed43/output_43.json"
  else
    PF="42=$PD/seed42/output_42.json 43=$PD/seed43/output_43.json"
  fi
  CUDA_VISIBLE_DEVICES=$GPU "$V/bin/python" "$P/scripts/bpo/judge_bpo.py" \
    --policy-files $PF \
    --reference-files "44=$P4/train_pool/generations/seed44/output_44.json" "45=$P4/train_pool/generations/seed45/output_45.json" \
    --judge-model-path "$JUDGE" --output "$SD/verdicts.jsonl" > "$SD/judge.log" 2>&1
  ;;
build)
  PD=$1
  if [ "$PD" = "-" ]; then
    PF="42=$P4/train_pool/generations/seed42/output_42.json 43=$P4/train_pool/generations/seed43/output_43.json"
  else
    PF="42=$PD/seed42/output_42.json 43=$PD/seed43/output_43.json"
  fi
  "$V/bin/python" "$P/scripts/bpo/build_bpo_pairs.py" --verdicts "$SD/verdicts.jsonl" \
    --policy-files $PF --out-dir "$SD/pool" --split-salt "bpo-stage$STAGE"
  ;;
precompute)
  PARENT=$1; GPU=$2
  CUDA_VISIBLE_DEVICES=$GPU "$V/bin/python" -m accelerate.commands.launch \
    --config_file "$P/accelerate_configs/single_gpu.yaml" --num_processes=1 --main_process_port=$((31300 + GPU)) \
    -m mnpo_scripts.precompute \
    --model_name_or_path "$BASE" --ref_model "$BASE" --history_paths "$PARENT" \
    --train_dir "$SD/pool/pairs_train.jsonl" --test_dir "$SD/pool/pairs_test.jsonl" \
    --output_dir "$SD/pool/logps" --per_device_train_batch_size 2 --per_device_eval_batch_size 2 \
    --max_length 2048 --max_prompt_length 1024 --apply_chat_template true \
    --auto_insert_empty_system_msg false --ronpo_target_mode none --report_to none \
    > "$SD/precompute.log" 2>&1
  ;;
train)
  ARM=$1; PARENT=$2; GPU=$3
  for mode in smoke full; do
    OUT=$SD/train/$ARM/$mode; mkdir -p "$OUT"
    [ -f "$OUT/all_results.json" ] && continue
    steps=20; [ "$mode" = full ] && steps=900
    cat > "$OUT/config.yaml" <<EOF
model_name_or_path: $PARENT
torch_dtype: null
attn_implementation: sdpa
dataset_mixer:
  ? $SD/pool/logps
  : 1.0
dataset_splits:
- train
- test
preprocessing_num_workers: 4
bf16: true
loss_type: ht_mnpo
eta: 0.0075
ht_target_column: bpo_target_$ARM
ht_target_scale: 1.0
max_history_t: 1
history_weights:
- 1.0
beta: 10.0
reference_anchor_weight: 0.05
preference_sft_weight: 0.005
learning_rate: 5.0e-07
lr_scheduler_type: cosine
warmup_ratio: 0.1
optim: adamw_torch
weight_decay: 0.0
max_grad_norm: 1.0
seed: 42
gradient_accumulation_steps: 16
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
num_train_epochs: 100
max_steps: $steps
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
max_length: 2048
max_prompt_length: 1024
do_eval: false
eval_strategy: 'no'
logging_steps: 5
log_level: info
save_strategy: 'no'
save_only_model: true
save_safetensors: true
push_to_hub: false
report_to:
- wandb
output_dir: $OUT
run_name: bpo-stage$STAGE-$ARM-$mode
EOF
    CUDA_VISIBLE_DEVICES=$GPU "$V/bin/python" -m accelerate.commands.launch \
      --config_file "$P/accelerate_configs/single_gpu.yaml" --num_processes=1 \
      --main_process_port=$((31400 + GPU)) -m mnpo_scripts.run_mnpo "$OUT/config.yaml" \
      > "$OUT/train.log" 2>&1
    [ -f "$OUT/all_results.json" ] || { echo "TRAIN_FAILED $ARM $mode"; exit 1; }
  done
  ;;
*) echo "unknown phase $PHASE" >&2; exit 2 ;;
esac
echo "PHASE_DONE $PHASE stage$STAGE $*"
