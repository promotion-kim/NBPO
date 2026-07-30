#!/usr/bin/env bash
# Decode one SafeRLHF arm on the shared eval prompts, convert to the Beaver input
# format, and score helpfulness (reward) + harmlessness (neg cost).
# Usage: nbpo_eval.sh ARM MODELDIR GPU
set -euo pipefail
ARM=$1; MODEL=$2; GPU=$3
P=/NHNHOME/AIPR/sjkim/MNPO_rev_20260710
V=/NHNHOME/AIPR/sjkim/venv_clean
R=/NHNHOME/AIPR/sjkim/nbpo_saferlhf_20260725
MAN=$R/eval/eval_prompts.jsonl
DEC=$P/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py
CACHE=/NHNHOME/AIPR/sjkim/cache/qwen25_table3
REWARD=$CACHE/models--PKU-Alignment--beaver-7b-v1.0-reward/snapshots/375cd6a9f0d7e339d2199b05ba129a4a8906596d
COST=$CACHE/models--PKU-Alignment--beaver-7b-v1.0-cost/snapshots/c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28
OUT=$R/eval/$ARM; mkdir -p $OUT
export PYTHONPATH=$P HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1
# 1. decode (greedy-ish, single sample per prompt)
if [ ! -s $OUT/gen.json ]; then
CUDA_VISIBLE_DEVICES=$GPU $V/bin/python $DEC --manifest $MAN --model $MODEL \
  --policy-name $ARM --probe none --output $OUT/gen.json --seed 42 \
  --temperature 0.7 --top-p 0.9 --max-new-tokens 512 --max-model-len 8192 \
  --gpu-memory-utilization 0.88 > $OUT/decode.log 2>&1
fi
# 2. convert generated_text -> all_generated_responses
$V/bin/python -c "import json;d=json.load(open('$OUT/gen.json'));[r.update({'all_generated_responses':[str(r['generated_text'])]}) for r in d];json.dump(d,open('$OUT/gen_conv.json','w'))"
# 3. Beaver reward (helpfulness) + cost (harmlessness = neg cost)
CUDA_VISIBLE_DEVICES=$GPU $V/bin/python $P/on_policy_data_gen/rm_beaver_reward.py \
  --model_name $REWARD --model_revision 375cd6a9f0d7e339d2199b05ba129a4a8906596d \
  --input_file $OUT/gen_conv.json --output_file $OUT/help.jsonl \
  --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 > $OUT/help.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU $V/bin/python $P/on_policy_data_gen/rm_beaver_cost.py \
  --model_name $COST --model_revision c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28 \
  --input_file $OUT/gen_conv.json --output_file $OUT/harm.jsonl \
  --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 > $OUT/harm.log 2>&1
echo "DONE $ARM" > $OUT/EVAL_DONE
