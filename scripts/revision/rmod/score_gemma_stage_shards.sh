#!/usr/bin/env bash
set -euo pipefail

WORK=${WORK:?set WORK}
SHARDS=${SHARDS:?comma-separated shard ids}
GPUS=${GPUS:?comma-separated GPU ids}
SCORE_SHARDS=${SCORE_SHARDS:-10}
SJ=/NHNHOME/AIPR/sjkim
P=$SJ/MNPO_rev_20260720
PY=$SJ/venv_clean/bin/python
CACHE=$SJ/baseline_repair_1p5b_20260714/cache/huggingface/hub
export PYTHONPATH=$P HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface
export HF_HUB_CACHE=$CACHE HUGGINGFACE_HUB_CACHE=$CACHE TOKENIZERS_PARALLELISM=false
IFS=, read -r -a shard <<< "$SHARDS"
IFS=, read -r -a gpu <<< "$GPUS"
(( ${#shard[@]} == ${#gpu[@]} ))
for split in train test; do
  while [[ ! -s "$WORK/pool/$split/merged/all_outputs.json" ]]; do sleep 5; done
  pids=()
  for i in "${!shard[@]}"; do
    CUDA_VISIBLE_DEVICES=${gpu[$i]} "$PY" -m on_policy_data_gen.rm_armo_multihead \
      --input_file "$WORK/pool/$split/merged/all_outputs.json" --output_dir "$WORK/scored" \
      --split "$split" --indices 6,7,8,9,10 \
      --names instruction_following,truthfulness,honesty,helpfulness,safety \
      --cache_dir "$CACHE" --batch_size 8 --sample_batch_size 32 \
      --num_shards "$SCORE_SHARDS" --shard_index "${shard[$i]}" \
      > "$WORK/pool/logs/score_${split}_shard${shard[$i]}.log" 2>&1 & pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done
