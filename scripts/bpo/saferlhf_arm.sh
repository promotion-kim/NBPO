#!/usr/bin/env bash
# Train one SafeRLHF stage-1 arm from base (Llama-3.1-8B) on the shared stage-1 pool.
# Usage: saferlhf_arm.sh ARM LOSS TARGETCOL GPU
#   ronpo-loss arms differ ONLY in TARGETCOL (target_nbpo_k0p1 / target_os_k0p1 / target_uniform)
set -euo pipefail
ARM=$1; LOSS=$2; TCOL=$3; GPU=$4
P=/NHNHOME/AIPR/sjkim/MNPO_rev_20260710
V=/NHNHOME/AIPR/sjkim/venv_clean
BASE=/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1/base_objective_screen/hf_ipv4/llama31
DS=/NHNHOME/AIPR/sjkim/maxmin_saferlhf_20260724/stage1_pool/precompute_nbpo2
ROOT=/NHNHOME/AIPR/sjkim/nbpo_saferlhf_20260725
OUT=$ROOT/$ARM; mkdir -p $OUT
export PYTHONPATH=$P HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cat > $OUT/config.yaml <<EOF
model_name_or_path: $BASE
torch_dtype: null
attn_implementation: sdpa
dataset_mixer:
  $DS: 1.0
dataset_splits:
- train
- test
preprocessing_num_workers: 4
bf16: true
loss_type: $LOSS
max_history_t: 1
history_weights:
- 1.0
ronpo_alpha: 1.0
ronpo_tau: 0.05
ronpo_target_column: $TCOL
eta: 0.0075
beta: 10.0
simpo_beta: 2.0
simpo_gamma: 1.0
dpo_beta: 0.01
learning_rate: 5.0e-7
lr_scheduler_type: cosine
warmup_ratio: 0.1
optim: adamw_torch
weight_decay: 0.0
max_grad_norm: 1.0
seed: 42
gradient_accumulation_steps: 8
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
num_train_epochs: 1
per_device_train_batch_size: 2
per_device_eval_batch_size: 2
max_length: 2048
max_prompt_length: 1800
do_eval: false
eval_strategy: 'no'
logging_steps: 20
save_strategy: 'no'
save_only_model: true
save_safetensors: true
push_to_hub: false
report_to:
- wandb
output_dir: $OUT
run_name: nbpo-saferlhf-$ARM
EOF
CUDA_VISIBLE_DEVICES=$GPU $V/bin/python -m accelerate.commands.launch \
  --config_file $P/accelerate_configs/single_gpu.yaml --num_processes=1 \
  --main_process_port=$((29600+GPU)) -m mnpo_scripts.run_mnpo $OUT/config.yaml > $OUT/train.log 2>&1
[ -f $OUT/model.safetensors ] && echo "DONE $ARM" > $OUT/DONE || echo "FAILED $ARM"
