#!/usr/bin/env bash
# Radar (Task 1) response pools on the 647 UltraFeedback test prompts.
# base gemma-2-2b-it: 16 samples (best-of-n pool; seed 42 = reference).
# downloadable baseline policies: 1 sample each. Scored later by ArmoRM 5 heads.
# GPUS = space-separated gpu ids on this host (default "0 1 3").
set -uo pipefail
SJ=/NHNHOME/AIPR/sjkim
PROJECT=$SJ/MNPO_rev_20260720
WORK=$SJ/rmod_20260720/radar
PY=$SJ/venv_clean/bin/python
export PYTHONPATH=$PROJECT
export HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TOKENIZERS_PARALLELISM=false
TE=$PROJECT/data/gemma2_ufb_part2_test.jsonl
read -ra G <<< "${GPUS:-0 1 3}"
mkdir -p $WORK/gens/logs
snap() { ls -d $HF_HUB_CACHE/models--Robust-Decoding--$1/snapshots/* 2>/dev/null | head -1; }

dec() { # gpu name model seeds
  local gpu=$1 name=$2 model=$3 seeds=$4
  CUDA_VISIBLE_DEVICES=$gpu $PY -u -m on_policy_data_gen.decode \
    --data_dir "$TE" --model "$model" --seeds $seeds \
    --output_dir "$WORK/gens/$name" --num_gpu 1 \
    --temperature 0.8 --top_p 0.95 --max_tokens 512 --batch_size 256 \
    --dtype bfloat16 --cache_dir "$HF_HUB_CACHE" \
    > "$WORK/gens/logs/$name.log" 2>&1
}

dec "${G[0]}" base_bon google/gemma-2-2b-it "42 13 21 79 100 7 11 23 47 53 61 71 83 97 3 5" & p0=$!
dec "${G[1]}" dpo_uniform "$(snap gemma22bit-hh-dpo-uniform-step60291)" 42 & p1=$!
dec "${G[2]:-${G[1]}}" grpo_uniform "$(snap gemma22bit-hh-grpo-uniform-step1000)" 42 & p2=$!
wait $p0 $p1 $p2
dec "${G[1]}" rmod_distill "$(snap gemma22bit-hh-RMODdistill_lr1e-5_3epochs_16kprompts)" 42
echo "[radar-pool] done at $(date -Is)"
