#!/usr/bin/env bash
# Decode one UFC arm on held-out UF prompts, score ArmoRM heads
# {helpfulness, instruction_following, honesty, conciseness=-verbosity}.
# Usage: uf_sft_eval.sh ARM MODEL GPU
set -euo pipefail
ARM=$1; MODEL=$2; GPU=$3
P=/NHNHOME/AIPR/sjkim/MNPO_rev_20260710
P2=/NHNHOME/AIPR/sjkim/MNPO_rev_20260720
V=/NHNHOME/AIPR/sjkim/venv_clean
R=/NHNHOME/AIPR/sjkim/nbpo_ufc_20260726
MAN=/NHNHOME/AIPR/sjkim/nbpo_uf5_20260725/eval_prompts_patched.jsonl
DEC=$P/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py
ARMO_CACHE=/NHNHOME/AIPR/sjkim/baseline_repair_1p5b_20260714/cache/huggingface/hub
OUT=$R/eval/$ARM; mkdir -p $OUT
export PYTHONPATH=$P2 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1
if [ ! -s $OUT/gen.json ]; then
CUDA_VISIBLE_DEVICES=$GPU $V/bin/python $DEC --manifest $MAN --model $MODEL \
  --policy-name $ARM --probe none --output $OUT/gen.json --seed 42 \
  --temperature 0.7 --top-p 0.9 --max-new-tokens 1024 --max-model-len 4096 \
  --gpu-memory-utilization 0.85 > $OUT/decode.log 2>&1
fi
$V/bin/python -c "import json;d=json.load(open('$OUT/gen.json'));[r.update({'all_generated_responses':[str(r['generated_text'])]}) for r in d];json.dump(d,open('$OUT/gen_conv.json','w'))"
AD=$ARMO_CACHE/models--RLHFlow--ArmoRM-Llama3-8B-v0.1
if [ ! -f $AD/refs/main ]; then snap=$(ls $AD/snapshots/ 2>/dev/null | head -1); [ -n "$snap" ] && mkdir -p $AD/refs && printf '%s' "$snap" > $AD/refs/main; fi
CUDA_VISIBLE_DEVICES=$GPU $V/bin/python $P2/on_policy_data_gen/rm_armo_multihead.py \
  --input_file $OUT/gen_conv.json --output_dir $OUT --split eval \
  --indices 9,6,8,4 --names helpfulness,instruction_following,honesty,conciseness \
  --negate_indices 4 --cache_dir $ARMO_CACHE > $OUT/score.log 2>&1
# rename eval_<name>.jsonl -> eval_<name>.jsonl already matches compute_uk_ir expectation
echo "DONE $ARM" > $OUT/EVAL_DONE
