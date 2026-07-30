#!/usr/bin/env bash
# Correct-oracle NBPO on HH-SFT: LLM-judge (Qwen3-32B) preference targets + ht_mnpo
# (MNPO) loss, then LLM-judge eval u_k. Trains unif/nbs/ks/maxmin from the SFT base.
# Assumes training verdicts already at $R/bpo/verdicts/shard*.jsonl. Usage: hh_bpo_run.sh
set -euo pipefail
P=/NHNHOME/AIPR/sjkim/MNPO_rev_20260710
V=/NHNHOME/AIPR/sjkim/venv_clean
BASE=/NHNHOME/AIPR/sjkim/sft_bases/zephyr-7b-sft-full
R=/NHNHOME/AIPR/sjkim/nbpo_hh_sft_20260726
J=/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1/flagship_20260712/cache/huggingface/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
DEC=$P/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py
MAN=$R/hh_test_prompts.jsonl
OBJ=helpfulness,harmlessness,humor
ETA=0.0075
E=$R/bpo
export PYTHONPATH=$P HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1
G=${1:-0}  # base GPU for serial steps

# 1. build 4-arm pairs from LLM-judge verdicts
cat $E/verdicts/shard*.jsonl > $E/verdicts_all.jsonl
$V/bin/python $P/scripts/bpo/build_bpo_pairs.py --verdicts $E/verdicts_all.jsonl \
  --policy-files 42=$R/gens/g42.json 43=$R/gens/g43.json \
  --out-dir $E/pool --split-salt hhbpo > $E/build.log 2>&1
# 2. precompute (ref=SFT, history=SFT)
CUDA_VISIBLE_DEVICES=$G $V/bin/python -m accelerate.commands.launch \
  --config_file $P/accelerate_configs/single_gpu.yaml --num_processes=1 --main_process_port=$((31500+G)) \
  -m mnpo_scripts.precompute --model_name_or_path $BASE --ref_model $BASE --history_paths $BASE \
  --train_dir $E/pool/pairs_train.jsonl --test_dir $E/pool/pairs_test.jsonl --output_dir $E/pre \
  --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --max_length 1536 --max_prompt_length 1024 \
  --apply_chat_template true --auto_insert_empty_system_msg false --ronpo_target_mode none --report_to none >> $E/precompute.log 2>&1
# 3. train unif/nbs/ks/maxmin (ht_mnpo, bpo_target_{arm})
g=0
for arm in unif nbs ks maxmin; do
  OUT=$E/train/$arm; mkdir -p $OUT
  cat > $OUT/config.yaml <<YAML
model_name_or_path: $BASE
attn_implementation: sdpa
dataset_mixer: {$E/pre: 1.0}
dataset_splits: [train, test]
bf16: true
loss_type: ht_mnpo
eta: $ETA
ht_target_column: bpo_target_$arm
ht_target_scale: 1.0
max_history_t: 1
history_weights: [1.0]
beta: 10.0
reference_anchor_weight: 0.05
preference_sft_weight: 0.005
learning_rate: 5.0e-7
lr_scheduler_type: cosine
warmup_ratio: 0.1
optim: adamw_torch
seed: 42
gradient_accumulation_steps: 8
gradient_checkpointing: true
gradient_checkpointing_kwargs: {use_reentrant: false}
num_train_epochs: 1
max_steps: 900
per_device_train_batch_size: 2
per_device_eval_batch_size: 2
max_length: 1536
max_prompt_length: 1024
do_eval: false
eval_strategy: 'no'
logging_steps: 50
save_strategy: 'no'
save_only_model: true
save_safetensors: true
report_to: [wandb]
output_dir: $OUT
run_name: hhbpo-$arm
YAML
  CUDA_VISIBLE_DEVICES=$g WANDB_PROJECT=nbpo WANDB_ENTITY=promotion-kim $V/bin/python -m accelerate.commands.launch \
    --config_file $P/accelerate_configs/single_gpu.yaml --num_processes=1 --main_process_port=$((31510+g)) \
    -m mnpo_scripts.run_mnpo $OUT/config.yaml > $OUT/train.log 2>&1 &
  g=$((g+1))
done
wait
echo BPO_TRAIN_DONE > $E/TRAIN_DONE
