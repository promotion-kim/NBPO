#!/usr/bin/env bash
set -euo pipefail

SJ=/NHNHOME/AIPR/sjkim
PROJECT=$SJ/MNPO_rev_20260720
REPO=$SJ/rmod_20260720/repo
OUT=$SJ/rmod_20260720/saferlhf_vf_disjoint_run
CKPT_LIST=$OUT/checkpoints.txt
while [[ ! -s $CKPT_LIST ]]; do
  sleep 30
done
CKPT=$(head -1 "$CKPT_LIST")
for _ in 1 2 3; do
  date -Is >> "$OUT/rank_probe_gpu1_prelaunch.txt"
  nvidia-smi -i 1 --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits >> "$OUT/rank_probe_gpu1_prelaunch.txt"
  sleep 2
done
export HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export PYTHONPATH=$REPO
CUDA_VISIBLE_DEVICES=1 $SJ/rmod_20260720/venv_rmod/bin/python \
  "$REPO/vf_rank_probe.py" \
  --data "$SJ/rmod_20260720/saferlhf_vf_data" --checkpoint "$CKPT" \
  --cache_dir "$HF_HUB_CACHE" --num_prompts 300 \
  > "$OUT/rank_probe.log" 2>&1
