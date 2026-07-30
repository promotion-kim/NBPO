#!/usr/bin/env bash
# Iterative HH stage-2 for one arm: decode its stage-1 policy (4 seeds) -> score 3 RMs
# -> build K=3 pairs (weights RE-ESTIMATED = primal-dual) -> precompute(parent=stage-1)
# -> add targets -> train stage-2 from the stage-1 policy. Usage: hh_iterate.sh ARM TCOL GPU
set -euo pipefail
ARM=$1; TCOL=$2; GPU=$3
P=/NHNHOME/AIPR/sjkim/MNPO_rev_20260710
V=/NHNHOME/AIPR/sjkim/venv_clean
BASE=/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1/base_objective_screen/hf_ipv4/llama31
R=/NHNHOME/AIPR/sjkim/nbpo_hh_20260726
S1=$R/s42/$ARM
MAN=$R/hh_train_prompts.jsonl
DEC=$P/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py
RMS=/NHNHOME/AIPR/sjkim/hh_rms
HELP=$(ls -d $RMS/models--Ray2333--gpt2-large-helpful-reward_model/snapshots/*/ | head -1)
HARM=$(ls -d $RMS/models--Ray2333--gpt2-large-harmless-reward_model/snapshots/*/ | head -1)
HUM=$(ls -d $RMS/models--mohameddhiab--humor-no-humor/snapshots/*/ | head -1)
O=$R/stage2/$ARM; mkdir -p $O/gens
export PYTHONPATH=$P HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1
for s in 42 43 44 45; do
  [ -s $O/gens/g$s.json ] && continue
  CUDA_VISIBLE_DEVICES=$GPU $V/bin/python $DEC --manifest $MAN --model $S1 --policy-name ${ARM}_s2_$s \
    --probe none --output $O/gens/g$s.json --seed $s --temperature 0.9 --top-p 0.95 \
    --max-new-tokens 512 --max-model-len 2048 --gpu-memory-utilization 0.85 >> $O/decode.log 2>&1
done
$V/bin/python - <<PY
import json
base={}
for s in [42,43,44,45]:
    for r in json.load(open("$O/gens/g%d.json"%s)):
        base.setdefault(str(r["prompt_id"]),{"prompt":r["prompt"],"prompt_id":r["prompt_id"],"all_generated_responses":[]})["all_generated_responses"].append(str(r["generated_text"]))
pool=[r for r in base.values() if len(r["all_generated_responses"])==4]
open("$O/pool.jsonl","w").write("\n".join(json.dumps(r,ensure_ascii=False) for r in pool)+"\n")
print("pool",len(pool))
PY
[ -s $O/sc_helpful.jsonl ]  || CUDA_VISIBLE_DEVICES=$GPU $V/bin/python $P/scripts/bpo/rm_hh.py --input_file $O/pool.jsonl --output_file $O/sc_helpful.jsonl  --model_path $HELP --kind reward --batch_size 16 >> $O/score.log 2>&1
[ -s $O/sc_harmless.jsonl ] || CUDA_VISIBLE_DEVICES=$GPU $V/bin/python $P/scripts/bpo/rm_hh.py --input_file $O/pool.jsonl --output_file $O/sc_harmless.jsonl --model_path $HARM --kind reward --batch_size 16 >> $O/score.log 2>&1
[ -s $O/sc_humor.jsonl ]    || CUDA_VISIBLE_DEVICES=$GPU $V/bin/python $P/scripts/bpo/rm_hh.py --input_file $O/pool.jsonl --output_file $O/sc_humor.jsonl    --model_path $HUM  --kind humor  --batch_size 32 >> $O/score.log 2>&1
$V/bin/python $P/scripts/bpo/build_kobj_pairs.py --scored helpful=$O/sc_helpful.jsonl harmless=$O/sc_harmless.jsonl humor=$O/sc_humor.jsonl \
  --train-output $O/pairs_train.jsonl --test-output $O/pairs_test.jsonl --pairs-per-prompt 6 --internal-test-prompts 100 --split-salt hh-s2-$ARM >> $O/build.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU $V/bin/python -m accelerate.commands.launch --config_file $P/accelerate_configs/single_gpu.yaml --num_processes=1 --main_process_port=$((30300+GPU)) \
  -m mnpo_scripts.precompute --model_name_or_path $BASE --ref_model $BASE --history_paths $S1 \
  --train_dir $O/pairs_train.jsonl --test_dir $O/pairs_test.jsonl --output_dir $O/pre \
  --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --max_length 1536 --max_prompt_length 1024 \
  --apply_chat_template true --auto_insert_empty_system_msg false --ronpo_target_mode none --report_to none >> $O/precompute.log 2>&1
$V/bin/python -m mnpo_scripts.build_os_ronpo_targets --input_dir $O/pre --output_dir $O/pre_t --kappas 0.1 >> $O/build.log 2>&1
OUT=$O/train; mkdir -p $OUT
cat > $OUT/config.yaml <<YAML
model_name_or_path: $S1
attn_implementation: sdpa
dataset_mixer: {$O/pre_t: 1.0}
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
max_steps: 900
per_device_train_batch_size: 2
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
run_name: nbpo-hh-s2-$ARM
YAML
CUDA_VISIBLE_DEVICES=$GPU $V/bin/python -m accelerate.commands.launch --config_file $P/accelerate_configs/single_gpu.yaml --num_processes=1 --main_process_port=$((30400+GPU)) -m mnpo_scripts.run_mnpo $OUT/config.yaml >> $OUT/train.log 2>&1
[ -f $OUT/model.safetensors ] && echo "S2_DONE $ARM" > $O/DONE || echo "S2_FAILED $ARM"
