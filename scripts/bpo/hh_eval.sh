#!/usr/bin/env bash
# Decode one HH arm on test prompts, score helpful/harmless/humor. Usage: hh_eval.sh ARM MODEL GPU
set -euo pipefail
ARM=$1; MODEL=$2; GPU=$3
P=/NHNHOME/AIPR/sjkim/MNPO_rev_20260710
V=/NHNHOME/AIPR/sjkim/venv_clean
R=/NHNHOME/AIPR/sjkim/nbpo_hh_20260726
MAN=$R/hh_test_prompts.jsonl
DEC=$P/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py
RMS=/NHNHOME/AIPR/sjkim/hh_rms
HELP=$(ls -d $RMS/models--Ray2333--gpt2-large-helpful-reward_model/snapshots/*/ | head -1)
HARM=$(ls -d $RMS/models--Ray2333--gpt2-large-harmless-reward_model/snapshots/*/ | head -1)
HUM=$(ls -d $RMS/models--mohameddhiab--humor-no-humor/snapshots/*/ | head -1)
OUT=$R/eval/$ARM; mkdir -p $OUT
export PYTHONPATH=$P HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1
if [ ! -s $OUT/gen.json ]; then
CUDA_VISIBLE_DEVICES=$GPU $V/bin/python $DEC --manifest $MAN --model $MODEL \
  --policy-name $ARM --probe none --output $OUT/gen.json --seed 42 \
  --temperature 0.7 --top-p 0.9 --max-new-tokens 512 --max-model-len 2048 \
  --gpu-memory-utilization 0.85 > $OUT/decode.log 2>&1
fi
$V/bin/python -c "import json;d=json.load(open('$OUT/gen.json'));open('$OUT/gen.jsonl','w').write('\n'.join(json.dumps({'prompt':str(r['prompt']),'prompt_id':r['prompt_id'],'all_generated_responses':[str(r['generated_text'])]},ensure_ascii=False) for r in d))"
CUDA_VISIBLE_DEVICES=$GPU $V/bin/python $P/scripts/bpo/rm_hh.py --input_file $OUT/gen.jsonl --output_file $OUT/eval_helpful.jsonl  --model_path $HELP --kind reward --batch_size 16 > $OUT/score.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU $V/bin/python $P/scripts/bpo/rm_hh.py --input_file $OUT/gen.jsonl --output_file $OUT/eval_harmless.jsonl --model_path $HARM --kind reward --batch_size 16 >> $OUT/score.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU $V/bin/python $P/scripts/bpo/rm_hh.py --input_file $OUT/gen.jsonl --output_file $OUT/eval_humor.jsonl    --model_path $HUM  --kind humor  --batch_size 32 >> $OUT/score.log 2>&1
echo "DONE $ARM" > $OUT/EVAL_DONE
