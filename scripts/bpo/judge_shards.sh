#!/usr/bin/env bash
# Launch 4 judge shards on GPUs 0-3 of THIS host. Usage: judge_shards.sh ROOT OFFSET
set -euo pipefail
ROOT=$1; OFFSET=$2
P=/NHNHOME/AIPR/sjkim/MNPO_rev_20260710
V=/NHNHOME/AIPR/sjkim/venv_clean
J=/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1/flagship_20260712/cache/huggingface/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
P4=$P/results/p4_8b_saferlhf_table4_20260717
SD=$ROOT/stage2
mkdir -p "$SD/verdicts"
export PYTHONPATH=$P HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for g in 0 1 2 3; do
  idx=$((OFFSET+g))
  tmux new-session -d -s jd$idx "CUDA_VISIBLE_DEVICES=$g $V/bin/python $P/scripts/bpo/judge_bpo.py \
    --policy-files 42=$SD/gens/seed42/output_42.json 43=$SD/gens/seed43/output_43.json \
    --reference-files 44=$P4/train_pool/generations/seed44/output_44.json 45=$P4/train_pool/generations/seed45/output_45.json \
    --judge-model-path $J --output $SD/verdicts/shard$idx.jsonl \
    --num-shards 8 --shard-index $idx > $SD/verdicts/judge_$idx.log 2>&1"
done
sleep 2; tmux ls | grep '^jd' || true
