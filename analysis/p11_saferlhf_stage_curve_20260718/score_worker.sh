#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 4 ]] || { echo "usage: $0 PROJECT OUTPUT GPU SHARD" >&2; exit 2; }
PROJECT=$1; OUT=$2; GPU=$3; SHARD=$4
VENV=$PROJECT/../venv_clean
ROOT=/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1
REWARD=$ROOT/flagship_20260712/cache/huggingface/models--PKU-Alignment--beaver-7b-v1.0-reward/snapshots/375cd6a9f0d7e339d2199b05ba129a4a8906596d
COST=$ROOT/flagship_20260712/cache/huggingface/models--PKU-Alignment--beaver-7b-v1.0-cost/snapshots/c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28
mkdir -p "$OUT/score_shards" "$OUT/logs/scores"
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PROJECT TOKENIZERS_PARALLELISM=false "$VENV/bin/python" "$PROJECT/on_policy_data_gen/rm_beaver_reward.py" \
  --model_name "$REWARD" --model_revision 375cd6a9f0d7e339d2199b05ba129a4a8906596d \
  --input_file "$OUT/shards/shard_${SHARD}.jsonl" --output_file "$OUT/score_shards/helpfulness_${SHARD}.jsonl" \
  --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 > "$OUT/logs/scores/helpfulness_${SHARD}_$(hostname).log" 2>&1
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PROJECT TOKENIZERS_PARALLELISM=false "$VENV/bin/python" "$PROJECT/on_policy_data_gen/rm_beaver_cost.py" \
  --model_name "$COST" --model_revision c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28 \
  --input_file "$OUT/shards/shard_${SHARD}.jsonl" --output_file "$OUT/score_shards/harmlessness_${SHARD}.jsonl" \
  --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 > "$OUT/logs/scores/harmlessness_${SHARD}_$(hostname).log" 2>&1
