#!/usr/bin/env bash
# HH-humor K=3 data build on one GPU: decode base (4 seeds) on train prompts,
# score helpful/harmless/humor, build pairs, precompute, add NBPO/OS/uniform targets.
# Usage: hh_build.sh GPU
set -euo pipefail
GPU=$1
P=/NHNHOME/AIPR/sjkim/MNPO_rev_20260710
V=/NHNHOME/AIPR/sjkim/venv_clean
BASE=/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1/base_objective_screen/hf_ipv4/llama31
R=/NHNHOME/AIPR/sjkim/nbpo_hh_20260726
MAN=$R/hh_train_prompts.jsonl
DEC=$P/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py
RMS=/NHNHOME/AIPR/sjkim/hh_rms
HELP=$(ls -d $RMS/models--Ray2333--gpt2-large-helpful-reward_model/snapshots/*/ | head -1)
HARM=$(ls -d $RMS/models--Ray2333--gpt2-large-harmless-reward_model/snapshots/*/ | head -1)
HUM=$(ls -d $RMS/models--mohameddhiab--humor-no-humor/snapshots/*/ | head -1)
export PYTHONPATH=$P HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1
mkdir -p $R/gens
# 1. decode 4 seeds (diverse pool)
for s in 42 43 44 45; do
  [ -s $R/gens/g$s.json ] && continue
  CUDA_VISIBLE_DEVICES=$GPU $V/bin/python $DEC --manifest $MAN --model $BASE \
    --policy-name hh_base_$s --probe none --output $R/gens/g$s.json --seed $s \
    --temperature 0.9 --top-p 0.95 --max-new-tokens 512 --max-model-len 2048 \
    --gpu-memory-utilization 0.85 >> $R/decode.log 2>&1
done
# 2. merge to pool of 4 -> jsonl for scoring
$V/bin/python - <<PY
import json
base={}
for s in [42,43,44,45]:
    for r in json.load(open("$R/gens/g%d.json"%s)):
        base.setdefault(str(r["prompt_id"]),{"prompt":r["prompt"],"prompt_id":r["prompt_id"],"all_generated_responses":[]})["all_generated_responses"].append(str(r["generated_text"]))
pool=[r for r in base.values() if len(r["all_generated_responses"])==4]
open("$R/pool.jsonl","w").write("\n".join(json.dumps(r,ensure_ascii=False) for r in pool)+"\n")
print("pool",len(pool))
PY
# 3. score 3 objectives with the shared scorer (consistent with eval)
[ -s $R/sc_helpful.jsonl ] || CUDA_VISIBLE_DEVICES=$GPU $V/bin/python $P/scripts/bpo/rm_hh.py --input_file $R/pool.jsonl --output_file $R/sc_helpful.jsonl  --model_path $HELP --kind reward --batch_size 16 >> $R/score.log 2>&1
[ -s $R/sc_harmless.jsonl ] || CUDA_VISIBLE_DEVICES=$GPU $V/bin/python $P/scripts/bpo/rm_hh.py --input_file $R/pool.jsonl --output_file $R/sc_harmless.jsonl --model_path $HARM --kind reward --batch_size 16 >> $R/score.log 2>&1
[ -s $R/sc_humor.jsonl ] || CUDA_VISIBLE_DEVICES=$GPU $V/bin/python $P/scripts/bpo/rm_hh.py --input_file $R/pool.jsonl --output_file $R/sc_humor.jsonl    --model_path $HUM  --kind humor  --batch_size 32 >> $R/score.log 2>&1
# 4. build K=3 pairs
$V/bin/python $P/scripts/bpo/build_kobj_pairs.py \
  --scored helpful=$R/sc_helpful.jsonl harmless=$R/sc_harmless.jsonl humor=$R/sc_humor.jsonl \
  --train-output $R/pairs_train.jsonl --test-output $R/pairs_test.jsonl \
  --pairs-per-prompt 6 --internal-test-prompts 100 --split-salt hh-k3 >> $R/build.log 2>&1
# 5. precompute (ref=base, history=base) + NBPO/OS/uniform targets
CUDA_VISIBLE_DEVICES=$GPU $V/bin/python -m accelerate.commands.launch \
  --config_file $P/accelerate_configs/single_gpu.yaml --num_processes=1 --main_process_port=$((30100+GPU)) \
  -m mnpo_scripts.precompute --model_name_or_path $BASE --ref_model $BASE --history_paths $BASE \
  --train_dir $R/pairs_train.jsonl --test_dir $R/pairs_test.jsonl --output_dir $R/pre \
  --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --max_length 1536 --max_prompt_length 1024 \
  --apply_chat_template true --auto_insert_empty_system_msg false --ronpo_target_mode none --report_to none >> $R/precompute.log 2>&1
$V/bin/python -m mnpo_scripts.build_os_ronpo_targets --input_dir $R/pre --output_dir $R/pre_t --kappas 0.1 >> $R/build.log 2>&1
echo HH_BUILD_DONE > $R/BUILD_DONE
