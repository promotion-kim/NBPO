#!/usr/bin/env bash
# HH BPO stage-2 (primal-dual) under the LLM judge, PVB mode: decode a stage-1 policy
# on the training prompts, judge it vs the SFT reference (policy-vs-base) per objective
# so the anchored surpluses are now non-zero, RE-ESTIMATE the bargaining weights, and
# train stage-2 unif/nbs/ks/maxmin from the stage-1 policy. Usage: hh_bpo_iter.sh PARENT_ARM
set -euo pipefail
PARENT_ARM=${1:-nbs}
P=/NHNHOME/AIPR/sjkim/MNPO_rev_20260710
V=/NHNHOME/AIPR/sjkim/venv_clean
BASE=/NHNHOME/AIPR/sjkim/sft_bases/zephyr-7b-sft-full
R=/NHNHOME/AIPR/sjkim/nbpo_hh_sft_20260726
J=/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1/flagship_20260712/cache/huggingface/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
DEC=$P/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py
MAN=$R/hh_train_prompts.jsonl
OBJ=helpfulness,harmlessness,humor
PARENT=$R/bpo/train/$PARENT_ARM
E=$R/bpo/s2
export PYTHONPATH=$P HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1
mkdir -p $E/gens

# 1. decode stage-1 policy on TRAIN prompts, seeds 42,43 (2 GPUs)
dec(){ local s=$1 g=$2; local out=$E/gens/seed$s/output_$s.json; mkdir -p "$(dirname "$out")"; [ -s "$out" ] && return
  CUDA_VISIBLE_DEVICES=$g $V/bin/python $DEC --manifest $MAN --model $PARENT --policy-name s2_$s \
    --probe none --output "$out" --seed $s --temperature 0.9 --top-p 0.95 --max-new-tokens 512 \
    --max-model-len 2048 --gpu-memory-utilization 0.85 > $E/dec_$s.log 2>&1; }
dec 42 1 & dec 43 2 & wait
echo "[s2] DECODE_DONE"

# 2. PVB judge: stage-1 policy (42,43) vs SFT reference base (44,45), sharded 4 GPUs
VD=$E/verdicts; mkdir -p $VD
for spec in "1 0" "2 1" "3 2"; do set -- $spec; g=$1; idx=$2
  tmux new-session -d -s js$idx "CUDA_VISIBLE_DEVICES=$g $V/bin/python $P/scripts/bpo/judge_bpo.py \
    --policy-files 42=$E/gens/seed42/output_42.json 43=$E/gens/seed43/output_43.json \
    --reference-files 44=$R/gens/g44.json 45=$R/gens/g45.json \
    --judge-model-path $J --objectives $OBJ --max-prompts 2500 \
    --output $VD/shard$idx.jsonl --num-shards 3 --shard-index $idx > $VD/j$idx.log 2>&1"
done
sleep 8; while tmux ls 2>/dev/null | grep -q '^js'; do sleep 20; done
cat $VD/shard*.jsonl > $E/verdicts_all.jsonl
echo "[s2] JUDGE_DONE"

# 3. build pairs PVB (anchored surplus vs reference -> re-estimated weights)
$V/bin/python $P/scripts/bpo/build_bpo_pairs.py --verdicts $E/verdicts_all.jsonl \
  --pair-mode pvb --policy-files 42=$E/gens/seed42/output_42.json 43=$E/gens/seed43/output_43.json \
  --base-files 44=$R/gens/g44.json 45=$R/gens/g45.json \
  --out-dir $E/pool --split-salt hhbpo-s2 > $E/build.log 2>&1
grep -A6 '"weights"' $E/pool/bpo_summary.json | head -20
# 4. precompute (parent = stage-1 policy)
CUDA_VISIBLE_DEVICES=1 $V/bin/python -m accelerate.commands.launch \
  --config_file $P/accelerate_configs/single_gpu.yaml --num_processes=1 --main_process_port=31560 \
  -m mnpo_scripts.precompute --model_name_or_path $BASE --ref_model $BASE --history_paths $PARENT \
  --train_dir $E/pool/pairs_train.jsonl --test_dir $E/pool/pairs_test.jsonl --output_dir $E/pre \
  --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --max_length 1536 --max_prompt_length 1024 \
  --apply_chat_template true --auto_insert_empty_system_msg false --ronpo_target_mode none --report_to none >> $E/precompute.log 2>&1
# 5. train stage-2 (decisive pair unif/nbs) from the stage-1 policy, GPU1/GPU2
for spec in "unif 1" "nbs 2"; do
  set -- $spec; arm=$1; gg=$2
  OUT=$E/train/$arm; mkdir -p $OUT
  cat > $OUT/config.yaml <<YAML
model_name_or_path: $PARENT
attn_implementation: sdpa
dataset_mixer: {$E/pre: 1.0}
dataset_splits: [train, test]
bf16: true
loss_type: ht_mnpo
eta: 0.0075
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
run_name: hhbpo-s2-$arm
YAML
  CUDA_VISIBLE_DEVICES=$gg WANDB_PROJECT=nbpo WANDB_ENTITY=promotion-kim $V/bin/python -m accelerate.commands.launch \
    --config_file $P/accelerate_configs/single_gpu.yaml --num_processes=1 --main_process_port=$((31570+gg)) \
    -m mnpo_scripts.run_mnpo $OUT/config.yaml > $OUT/train.log 2>&1 &
done
wait
echo BPO_S2_TRAIN_DONE > $E/TRAIN_DONE
