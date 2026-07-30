#!/usr/bin/env bash
# UF BPO stage-2 under the LLM judge with STRONG ronpo loss (the fix that separated HH):
# decode UF stage-1 nbs on train prompts, PVB-judge vs SFT reference (4 attributes) to
# re-estimate imbalanced bargaining weights, then train unif/nbs/ks/maxmin from the
# stage-1 policy with loss_type=ronpo (beta10). Runs on polymer 4 GPUs. Usage: uf_bpo_iter_ronpo.sh
set -uo pipefail
P=/NHNHOME/AIPR/sjkim/MNPO_rev_20260710
V=/NHNHOME/AIPR/sjkim/venv_clean
BASE=/NHNHOME/AIPR/sjkim/sft_bases/zephyr-7b-sft-full
R=/NHNHOME/AIPR/sjkim/nbpo_ufc_20260726
J=/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1/flagship_20260712/cache/huggingface/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
DEC=$P/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py
MAN=$R/uf_train_prompts.jsonl
OBJ=helpfulness,instruction_following,honesty,conciseness
PARENT=$R/bpo/train/nbs
E=$R/bpo/s2
export PYTHONPATH=$P HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1
mkdir -p $E/gens
# 1. decode stage-1 nbs on train prompts (seed42,43), GPU0,1
for gi in "42 0" "43 1"; do set -- $gi; s=$1; g=$2; o=$E/gens/seed$s/output_$s.json; mkdir -p "$(dirname "$o")"
  [ -s "$o" ] || CUDA_VISIBLE_DEVICES=$g $V/bin/python $DEC --manifest $MAN --model $PARENT --policy-name s2_$s \
    --probe none --output "$o" --seed $s --temperature 0.9 --top-p 0.95 --max-new-tokens 512 --max-model-len 4096 \
    --gpu-memory-utilization 0.85 > $E/dec_$s.log 2>&1 &
done; wait
# 2. PVB judge policy(42,43) vs SFT base(44,45), 4 obj, 4 shards GPU0-3
VD=$E/verdicts; mkdir -p $VD
for gi in "0 0" "1 1" "2 2" "3 3"; do set -- $gi; g=$1; ix=$2
  CUDA_VISIBLE_DEVICES=$g $V/bin/python $P/scripts/bpo/judge_bpo.py \
    --policy-files 42=$E/gens/seed42/output_42.json 43=$E/gens/seed43/output_43.json \
    --reference-files 44=$R/gens/g44.json 45=$R/gens/g45.json \
    --judge-model-path $J --objectives $OBJ --max-prompts 2500 \
    --output $VD/shard$ix.jsonl --num-shards 4 --shard-index $ix > $VD/j$ix.log 2>&1 &
done; wait
cat $VD/shard*.jsonl > $E/verdicts_all.jsonl
# 3. build pairs PVB (re-estimate weights)
$V/bin/python $P/scripts/bpo/build_bpo_pairs.py --verdicts $E/verdicts_all.jsonl \
  --pair-mode pvb --policy-files 42=$E/gens/seed42/output_42.json 43=$E/gens/seed43/output_43.json \
  --base-files 44=$R/gens/g44.json 45=$R/gens/g45.json --out-dir $E/pool --split-salt ufbpo-s2 > $E/build.log 2>&1
python3 -c "import json;w=json.load(open('$E/pool/bpo_summary.json'));print('nbs',{k:round(v,3) for k,v in w['weights']['nbs'].items()});print('surplus',{k:round(v,3) for k,v in (w.get('mean_surplus') or w.get('surplus') or {}).items()})"
# 4. precompute (parent = stage-1 nbs)
CUDA_VISIBLE_DEVICES=0 $V/bin/python -m accelerate.commands.launch \
  --config_file $P/accelerate_configs/single_gpu.yaml --num_processes=1 --main_process_port=31680 \
  -m mnpo_scripts.precompute --model_name_or_path $BASE --ref_model $BASE --history_paths $PARENT \
  --train_dir $E/pool/pairs_train.jsonl --test_dir $E/pool/pairs_test.jsonl --output_dir $E/pre \
  --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --max_length 2048 --max_prompt_length 1024 \
  --apply_chat_template true --auto_insert_empty_system_msg false --ronpo_target_mode none --report_to none >> $E/precompute.log 2>&1
# 5. train unif/nbs/ks/maxmin STRONG ronpo from stage-1 nbs, GPU0-3
g=0
for arm in unif nbs ks maxmin; do
  OUT=$E/train/$arm; mkdir -p $OUT
  cat > $OUT/config.yaml <<YAML
model_name_or_path: $PARENT
attn_implementation: sdpa
dataset_mixer: {$E/pre: 1.0}
dataset_splits: [train, test]
bf16: true
loss_type: ronpo
ronpo_alpha: 1.0
ronpo_tau: 0.05
ronpo_target_column: bpo_target_$arm
max_history_t: 1
history_weights: [1.0]
beta: 10.0
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
max_length: 2048
max_prompt_length: 1024
do_eval: false
save_strategy: 'no'
save_only_model: true
save_safetensors: true
report_to: [wandb]
output_dir: $OUT
run_name: ufbpo-s2r-$arm
YAML
  CUDA_VISIBLE_DEVICES=$g WANDB_PROJECT=nbpo WANDB_ENTITY=promotion-kim $V/bin/python -m accelerate.commands.launch \
    --config_file $P/accelerate_configs/single_gpu.yaml --num_processes=1 --main_process_port=$((31690+g)) \
    -m mnpo_scripts.run_mnpo $OUT/config.yaml > $OUT/train.log 2>&1 &
  g=$((g+1))
done
wait
echo DONE > $E/TRAIN_DONE
