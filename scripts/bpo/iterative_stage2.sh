#!/usr/bin/env bash
# Iterative NBPO/RONPO stage-2 for one arm: decode stage-1 policy (4 seeds) ->
# Beaver score -> build_shared_pairs -> precompute -> add target cols (weights
# RE-ESTIMATED from the stage-2 pool = the primal-dual update) -> train stage-2.
# Usage: iterative_stage2.sh ARM TCOL GPU
set -euo pipefail
ARM=$1; TCOL=$2; GPU=$3
P=/NHNHOME/AIPR/sjkim/MNPO_rev_20260710
V=/NHNHOME/AIPR/sjkim/venv_clean
BASE=/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1/base_objective_screen/hf_ipv4/llama31
R=/NHNHOME/AIPR/sjkim/nbpo_saferlhf_20260725
S1=$R/$ARM                                      # stage-1 policy (parent)
MAN=$R/eval/stage2_prompts.jsonl                # 1200 train_conflict prompts
DEC=$P/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py
CACHE=/NHNHOME/AIPR/sjkim/cache/qwen25_table3
REWARD=$CACHE/models--PKU-Alignment--beaver-7b-v1.0-reward/snapshots/375cd6a9f0d7e339d2199b05ba129a4a8906596d
COST=$CACHE/models--PKU-Alignment--beaver-7b-v1.0-cost/snapshots/c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28
O=$R/stage2/$ARM; mkdir -p $O/gens
export PYTHONPATH=$P HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1

# 1. decode 4 seeds -> response pool of 4 per prompt
for s in 42 43 44 45; do
  [ -s $O/gens/g$s.json ] && continue
  CUDA_VISIBLE_DEVICES=$GPU $V/bin/python $DEC --manifest $MAN --model $S1 \
    --policy-name ${ARM}_s2_$s --probe none --output $O/gens/g$s.json --seed $s \
    --temperature 0.9 --top-p 0.95 --max-new-tokens 512 --max-model-len 8192 \
    --gpu-memory-utilization 0.88 >> $O/decode.log 2>&1
done
# 2. merge into pool {prompt, all_generated_responses:[4]}
$V/bin/python - <<PY
import json
seeds=[42,43,44,45]; base={}
for s in seeds:
    for r in json.load(open("$O/gens/g%d.json"%s)):
        base.setdefault(str(r["prompt_id"]),{"prompt":r["prompt"],"prompt_id":r["prompt_id"],"all_generated_responses":[]})["all_generated_responses"].append(str(r["generated_text"]))
pool=[r for r in base.values() if len(r["all_generated_responses"])==4]
json.dump(pool,open("$O/pool.json","w")); print("pool",len(pool))
PY
# 3. Beaver score the 4-response pool (skip if already scored)
if [ ! -s $O/help.jsonl ] || [ ! -s $O/harm.jsonl ]; then
CUDA_VISIBLE_DEVICES=$GPU $V/bin/python $P/on_policy_data_gen/rm_beaver_reward.py \
  --model_name $REWARD --model_revision 375cd6a9f0d7e339d2199b05ba129a4a8906596d \
  --input_file $O/pool.json --output_file $O/help.jsonl --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 >> $O/score.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU $V/bin/python $P/on_policy_data_gen/rm_beaver_cost.py \
  --model_name $COST --model_revision c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28 \
  --input_file $O/pool.json --output_file $O/harm.jsonl --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 >> $O/score.log 2>&1
fi
# 4. build stage-2 pairs (self-play pool of 4)
$V/bin/python $P/scripts/bpo/build_shared_pairs.py \
  --helpfulness $O/help.jsonl --harmlessness $O/harm.jsonl \
  --train-output $O/pairs_train.jsonl --test-output $O/pairs_test.jsonl \
  --summary $O/pairs_summary.json --pairs-per-prompt 3 --expected-responses 4 \
  --expected-prompts 0 >> $O/build.log 2>&1
# 5. precompute logps (ref=base, history=stage-1 policy)
CUDA_VISIBLE_DEVICES=$GPU $V/bin/python -m accelerate.commands.launch \
  --config_file $P/accelerate_configs/single_gpu.yaml --num_processes=1 --main_process_port=$((29700+GPU)) \
  -m mnpo_scripts.precompute --model_name_or_path $BASE --ref_model $BASE --history_paths $S1 \
  --train_dir $O/pairs_train.jsonl --test_dir $O/pairs_test.jsonl --output_dir $O/pre \
  --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --max_length 2048 --max_prompt_length 1800 \
  --apply_chat_template true --auto_insert_empty_system_msg false --ronpo_target_mode none --report_to none >> $O/precompute.log 2>&1
# 6. add target columns with RE-ESTIMATED bargaining weights (primal-dual update)
$V/bin/python -m mnpo_scripts.build_os_ronpo_targets --input_dir $O/pre --output_dir $O/pre_t --kappas 0.1 >> $O/build.log 2>&1
# 7. train stage-2 from stage-1 policy
OUT=$O/train; mkdir -p $OUT
cat > $OUT/config.yaml <<EOF
model_name_or_path: $S1
attn_implementation: sdpa
dataset_mixer:
  $O/pre_t: 1.0
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
seed: 42
gradient_accumulation_steps: 8
gradient_checkpointing: true
gradient_checkpointing_kwargs: {use_reentrant: false}
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
report_to: [wandb]
output_dir: $OUT
run_name: nbpo-s2-$ARM
EOF
CUDA_VISIBLE_DEVICES=$GPU $V/bin/python -m accelerate.commands.launch \
  --config_file $P/accelerate_configs/single_gpu.yaml --num_processes=1 --main_process_port=$((29800+GPU)) \
  -m mnpo_scripts.run_mnpo $OUT/config.yaml >> $OUT/train.log 2>&1
[ -f $OUT/model.safetensors ] && echo "S2_DONE $ARM" > $O/DONE || echo "S2_FAILED $ARM"
