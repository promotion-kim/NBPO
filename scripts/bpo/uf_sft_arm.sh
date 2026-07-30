#!/usr/bin/env bash
# Train one UFC arm from the SFT base on precomputed targets. Like hh_arm.sh but
# max_length 2048 to match the UF precompute. Usage: uf_sft_arm.sh ARM TCOL GPU [SEED]
set -euo pipefail
ARM=$1; TCOL=$2; GPU=$3; SEED=${4:-42}
P=/NHNHOME/AIPR/sjkim/MNPO_rev_20260710
V=/NHNHOME/AIPR/sjkim/venv_clean
BASE=/NHNHOME/AIPR/sjkim/sft_bases/zephyr-7b-sft-full
R=/NHNHOME/AIPR/sjkim/nbpo_ufc_20260726
DS=$R/pre_t
OUT=$R/s$SEED/$ARM; mkdir -p $OUT
export PYTHONPATH=$P HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cat > $OUT/config.yaml <<YAML
model_name_or_path: $BASE
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
max_length: 2048
max_prompt_length: 1024
do_eval: false
eval_strategy: 'no'
logging_steps: 50
save_strategy: 'no'
save_only_model: true
save_safetensors: true
report_to: [wandb]
output_dir: $OUT
run_name: nbpo-ufc-$ARM-s$SEED
YAML
CUDA_VISIBLE_DEVICES=$GPU $V/bin/python -m accelerate.commands.launch \
  --config_file $P/accelerate_configs/single_gpu.yaml --num_processes=1 \
  --main_process_port=$((30500+GPU)) -m mnpo_scripts.run_mnpo $OUT/config.yaml > $OUT/train.log 2>&1
[ -f $OUT/model.safetensors ] && echo "DONE $ARM" > $OUT/DONE || echo "FAILED $ARM"
