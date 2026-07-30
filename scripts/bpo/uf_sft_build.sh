#!/usr/bin/env bash
# UltraFeedback K=4 build from the pre-alignment SFT base, scored by ArmoRM heads
# {helpfulness, instruction_following, honesty, conciseness=-verbosity}. conciseness is
# the genuinely-conflicting axis (it opposes the length-favoring quality heads), so the
# bargaining problem is non-degenerate with real conflict. Usage: uf_sft_build.sh GPU
set -euo pipefail
GPU=$1
P=/NHNHOME/AIPR/sjkim/MNPO_rev_20260710
P2=/NHNHOME/AIPR/sjkim/MNPO_rev_20260720
V=/NHNHOME/AIPR/sjkim/venv_clean
BASE=/NHNHOME/AIPR/sjkim/sft_bases/zephyr-7b-sft-full
R=/NHNHOME/AIPR/sjkim/nbpo_ufc_20260726
MAN=$R/uf_train_prompts.jsonl
DEC=$P/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py
ARMO_CACHE=/NHNHOME/AIPR/sjkim/baseline_repair_1p5b_20260714/cache/huggingface/hub
export PYTHONPATH=$P2 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1
mkdir -p $R/gens
# 1. decode 4 seeds from the SFT base
for s in 42 43 44 45; do
  [ -s $R/gens/g$s.json ] && continue
  CUDA_VISIBLE_DEVICES=$GPU $V/bin/python $DEC --manifest $MAN --model $BASE \
    --policy-name uf_sft_$s --probe none --output $R/gens/g$s.json --seed $s \
    --temperature 0.9 --top-p 0.95 --max-new-tokens 1024 --max-model-len 4096 \
    --gpu-memory-utilization 0.85 >> $R/decode.log 2>&1
done
# 2. merge to pool of 4
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
# 3. ArmoRM multi-head score (idx 9 helpfulness, 6 instruction_following, 8 honesty, 4 verbosity->negate=conciseness)
AD=$ARMO_CACHE/models--RLHFlow--ArmoRM-Llama3-8B-v0.1
if [ ! -f $AD/refs/main ]; then snap=$(ls $AD/snapshots/ 2>/dev/null | head -1); [ -n "$snap" ] && mkdir -p $AD/refs && printf '%s' "$snap" > $AD/refs/main; fi
[ -s $R/scored/train_conciseness.jsonl ] || CUDA_VISIBLE_DEVICES=$GPU $V/bin/python $P2/on_policy_data_gen/rm_armo_multihead.py \
  --input_file $R/pool.jsonl --output_dir $R/scored --split train \
  --indices 9,6,8,4 --names helpfulness,instruction_following,honesty,conciseness \
  --negate_indices 4 --cache_dir $ARMO_CACHE >> $R/score.log 2>&1
# 4. build K=4 pairs
$V/bin/python $P/scripts/bpo/build_kobj_pairs.py \
  --scored helpfulness=$R/scored/train_helpfulness.jsonl instruction_following=$R/scored/train_instruction_following.jsonl honesty=$R/scored/train_honesty.jsonl conciseness=$R/scored/train_conciseness.jsonl \
  --train-output $R/pairs_train.jsonl --test-output $R/pairs_test.jsonl \
  --pairs-per-prompt 6 --internal-test-prompts 100 --split-salt ufc-k4 >> $R/build.log 2>&1
# 5. precompute (ref=SFT base) + targets
CUDA_VISIBLE_DEVICES=$GPU $V/bin/python -m accelerate.commands.launch \
  --config_file $P/accelerate_configs/single_gpu.yaml --num_processes=1 --main_process_port=$((30160+GPU)) \
  -m mnpo_scripts.precompute --model_name_or_path $BASE --ref_model $BASE --history_paths $BASE \
  --train_dir $R/pairs_train.jsonl --test_dir $R/pairs_test.jsonl --output_dir $R/pre \
  --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --max_length 2048 --max_prompt_length 1024 \
  --apply_chat_template true --auto_insert_empty_system_msg false --ronpo_target_mode none --report_to none >> $R/precompute.log 2>&1
$V/bin/python -m mnpo_scripts.build_os_ronpo_targets --input_dir $R/pre --output_dir $R/pre_t --kappas 0.1 >> $R/build.log 2>&1
echo UF_SFT_BUILD_DONE > $R/BUILD_DONE
