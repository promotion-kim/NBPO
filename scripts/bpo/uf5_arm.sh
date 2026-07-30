#!/usr/bin/env bash
# Train one UltraFeedback-5obj arm from the INPO warmstart on the shared RONPO
# precomputed pool (5 ArmoRM heads). Arms differ ONLY in the target column.
# Usage: uf5_arm.sh ARM TCOL GPU [SEED]
set -euo pipefail
ARM=$1; TCOL=$2; GPU=$3; SEED=${4:-42}
P=/NHNHOME/AIPR/sjkim/MNPO_rev_20260710
V=/NHNHOME/AIPR/sjkim/venv_clean
PARENT=/NHNHOME/AIPR/sjkim/uf5_inpo_warmstart_robustft_20260724/train/inpo
DS=/NHNHOME/AIPR/sjkim/nbpo_uf5_20260725/pre_nbpo
ROOT=/NHNHOME/AIPR/sjkim/nbpo_uf5_20260725
OUT=$ROOT/s$SEED/$ARM; mkdir -p $OUT
export PYTHONPATH=$P HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cat > $OUT/config.yaml <<EOF
model_name_or_path: $PARENT
attn_implementation: sdpa
dataset_mixer:
  $DS: 1.0
dataset_splits: [train, test]
bf16: true
loss_type: ronpo
max_history_t: 1
history_weights: [1.0]
ronpo_alpha: 1.0
ronpo_tau: 0.05
ronpo_target_column: $TCOL
beta: 10.0
learning_rate: 1.0e-6
lr_scheduler_type: cosine
warmup_ratio: 0.1
optim: adamw_torch
seed: $SEED
gradient_accumulation_steps: 8
gradient_checkpointing: true
gradient_checkpointing_kwargs: {use_reentrant: false}
num_train_epochs: 1
max_steps: 800
per_device_train_batch_size: 2
per_device_eval_batch_size: 2
max_length: 2048
max_prompt_length: 1800
do_eval: false
eval_strategy: 'no'
logging_steps: 50
save_strategy: 'no'
save_only_model: true
save_safetensors: true
report_to: [wandb]
output_dir: $OUT
run_name: nbpo-uf5-$ARM-s$SEED
EOF
CUDA_VISIBLE_DEVICES=$GPU $V/bin/python -m accelerate.commands.launch \
  --config_file $P/accelerate_configs/single_gpu.yaml --num_processes=1 \
  --main_process_port=$((29900+GPU)) -m mnpo_scripts.run_mnpo $OUT/config.yaml > $OUT/train.log 2>&1
[ -f $OUT/model.safetensors ] && echo "DONE $ARM" > $OUT/DONE || echo "FAILED $ARM"
