#!/usr/bin/env bash
# HT-MNPO baseline (heterogeneous Multiplayer Nash PO): squared loss between the
# policy-vs-opponent log-ratio and eta*ht_target, ht_target = mean per-objective
# normalized reward gap (added to the DS as an `ht_target` column). Single history
# opponent (SFT). Usage: htmnpo_arm.sh GPU MAXLEN [SEED]  env: BASE R DS RUNTAG WANDB_*.
set -euo pipefail
GPU=$1; MAXLEN=${2:-1536}; SEED=${3:-42}
P=/NHNHOME/AIPR/sjkim/MNPO_rev_20260710
V=/NHNHOME/AIPR/sjkim/venv_clean
BASE=${BASE:?}; R=${R:?}; DS=${DS:?}
OUT=$R/s$SEED/htmnpo; mkdir -p $OUT
export PYTHONPATH=$P HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cat > $OUT/config.yaml <<YAML
model_name_or_path: $BASE
attn_implementation: sdpa
dataset_mixer:
  $DS: 1.0
dataset_splits: [train, test]
bf16: true
loss_type: ht_mnpo
eta: 0.0075
ratio: 0.3333
max_history_t: 1
history_weights: [1.0]
beta: 10.0
ht_target_column: ht_target
ht_target_scale: 1.0
learning_rate: 5.0e-7
lr_scheduler_type: cosine
warmup_ratio: 0.1
optim: adamw_torch
seed: $SEED
gradient_accumulation_steps: 8
gradient_checkpointing: true
gradient_checkpointing_kwargs: {use_reentrant: false}
num_train_epochs: 1
max_steps: 900
per_device_train_batch_size: 2
per_device_eval_batch_size: 2
max_length: $MAXLEN
max_prompt_length: 1024
do_eval: false
eval_strategy: 'no'
logging_steps: 50
save_strategy: 'no'
save_only_model: true
save_safetensors: true
report_to: [wandb]
output_dir: $OUT
run_name: ${RUNTAG:-nbpo}-htmnpo-s$SEED
YAML
CUDA_VISIBLE_DEVICES=$GPU $V/bin/python -m accelerate.commands.launch \
  --config_file $P/accelerate_configs/single_gpu.yaml --num_processes=1 \
  --main_process_port=$((30800+GPU)) -m mnpo_scripts.run_mnpo $OUT/config.yaml > $OUT/train.log 2>&1
[ -f $OUT/model.safetensors ] && echo "DONE htmnpo" > $OUT/DONE || echo "FAILED htmnpo"
