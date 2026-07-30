#!/usr/bin/env bash
# Judge arbitrary policy gens vs base ref, sharded on GPUs 0-3 of THIS host.
# Usage: judge_gen.sh GENS_DIR VERD_DIR OFFSET NUM_SHARDS
#   policy = GENS_DIR/seed{42,43}/output_{42,43}.json ; ref = base seed44/45
#   writes VERD_DIR/shard{OFFSET..OFFSET+3}.jsonl with --num-shards NUM_SHARDS
set -euo pipefail
GENS=$1; VERD=$2; OFFSET=$3; NS=$4
OBJ=${OBJ:-helpfulness,harmlessness}
P=/NHNHOME/AIPR/sjkim/MNPO_rev_20260710
V=/NHNHOME/AIPR/sjkim/venv_clean
J=/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1/flagship_20260712/cache/huggingface/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
P4=$P/results/p4_8b_saferlhf_table4_20260717
mkdir -p "$VERD"
export PYTHONPATH=$P HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for g in 0 1 2 3; do
  idx=$((OFFSET+g)); [ $idx -ge $NS ] && break
  tmux new-session -d -s jg$idx "CUDA_VISIBLE_DEVICES=$g $V/bin/python $P/scripts/bpo/judge_bpo.py \
    --policy-files 42=$GENS/seed42/output_42.json 43=$GENS/seed43/output_43.json \
    --reference-files 44=$P4/train_pool/generations/seed44/output_44.json 45=$P4/train_pool/generations/seed45/output_45.json \
    --judge-model-path $J --output $VERD/shard$idx.jsonl --objectives $OBJ \
    --num-shards $NS --shard-index $idx > $VERD/judge_$idx.log 2>&1"
done
sleep 2; tmux ls 2>/dev/null | grep '^jg' || true
