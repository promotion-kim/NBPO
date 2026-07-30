#!/usr/bin/env bash
set -euo pipefail
START=${1:?start shard}; END=${2:?end shard}; GPUS=${GPUS:-0,1,2,3}
SJ=/NHNHOME/AIPR/sjkim; P=$SJ/MNPO_rev_20260720; ROOT=$SJ/ronpo_uf5_combined_fresh_20260723
PY=$SJ/venv_clean/bin/python; CACHE=$SJ/baseline_repair_1p5b_20260714/cache/huggingface/hub
IFS=, read -r -a GPU <<< "$GPUS"
export PYTHONPATH=$P HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface
export HF_HUB_CACHE=$CACHE HUGGINGFACE_HUB_CACHE=$CACHE TOKENIZERS_PARALLELISM=false MNPO_DISABLE_APEX=1 NCCL_SOCKET_IFNAME=lo
while [[ ! -s "$ROOT/joint/joint_pool.READY" ]]; do sleep 20; done
mkdir -p "$ROOT/joint/scores" "$ROOT/joint/logs"
pids=(); local_index=0
for shard in $(seq "$START" "$END"); do
  CUDA_VISIBLE_DEVICES=${GPU[$local_index]} "$PY" -m on_policy_data_gen.rm_armo_multihead \
    --input_file "$ROOT/joint/joint_pool.json" --output_dir "$ROOT/joint/scores" --split joint \
    --indices 6,7,8,9,10 --names instruction_following,truthfulness,honesty,helpfulness,safety \
    --cache_dir "$CACHE" --batch_size 16 --sample_batch_size 1 --num_shards 6 --shard_index "$shard" \
    > "$ROOT/joint/logs/score_shard${shard}.log" 2>&1 & pids+=("$!")
  local_index=$((local_index+1))
done
for pid in "${pids[@]}"; do wait "$pid"; done
date -Is > "$ROOT/joint/SCORES_${START}_${END}_COMPLETE"
