#!/usr/bin/env bash
# Decode trained stage-2 arms for eval. Usage: decode_eval.sh ROOT arm:seed:gpu [arm:seed:gpu ...]
#   e.g. decode_eval.sh $B unif:42:0 unif:43:1 nbs:42:2 nbs:43:3
set -euo pipefail
ROOT=$1; shift
P=/NHNHOME/AIPR/sjkim/MNPO_rev_20260710
V=/NHNHOME/AIPR/sjkim/venv_clean
P4=$P/results/p4_8b_saferlhf_table4_20260717
SD=$ROOT/stage2; ED=$SD/eval
export PYTHONPATH=$P HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1
for spec in "$@"; do
  IFS=: read -r arm s g <<<"$spec"
  out=$ED/$arm/gens/seed$s/output_$s.json; mkdir -p "$(dirname "$out")"
  [ -s "$out" ] && continue
  tmux new-session -d -s ee_${arm}_$s "CUDA_VISIBLE_DEVICES=$g $V/bin/python $P/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py \
    --manifest $P4/dataset_manifest/train_conflict.jsonl --model $SD/train/$arm/full \
    --policy-name eval_${arm}_$s --probe none --output $out --seed $s \
    --temperature 0.7 --top-p 0.9 --max-new-tokens 512 --max-model-len 8192 --gpu-memory-utilization 0.88 \
    > $ED/$arm/decode_$s.log 2>&1"
done
sleep 2; tmux ls 2>/dev/null | grep '^ee_' || true
