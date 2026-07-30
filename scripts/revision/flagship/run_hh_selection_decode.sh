#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/venv_clean}
EXP=${EXP:-$PROJECT/results/p1_8b_hh_selection_20260716}
CACHE=${CACHE:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/revision_qwen3_8b/full_iter1/flagship_20260712/cache/huggingface/hub}
MANIFEST=$EXP/dataset_manifest/validation.jsonl
DECODER=$PROJECT/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py
mkdir -p "$EXP/generations" "$EXP/logs/decode"
source "$VENV/bin/activate"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_CUDNN_SDPA_ENABLED=0
export WANDB_MODE=disabled

BASE=$CACHE/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
WEAK=${CACHE%/hub}/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
RONPO_FULL=$CACHE/models--promotion--qwen3-8b-aaai27-flagship-ronpo-full-expect-s42/snapshots/17a1f1171627d43257182277e011e9b7b602ea53
RONPO_K=$CACHE/models--promotion--qwen3-8b-aaai27-flagship-ronpo-k-only-s42/snapshots/b8fdba53b2310b8b1f40079340138c3a5622df9f
IPO=$CACHE/models--promotion--qwen3-8b-aaai27-flagship-ipo-s42/snapshots/cdb29724dbf62b13828e53463846d7449a25bc10
SIMPO=$CACHE/models--promotion--qwen3-8b-aaai27-flagship-simpo-s42/snapshots/7d9fa7b7c38737775af5e705c2da89000fb3b85f
SPPO=$CACHE/models--promotion--qwen3-8b-aaai27-flagship-sppo-avg-s42/snapshots/3e0da986b4fffc05d12ac7a069ad2b875cbc1dd7
INPO=$CACHE/models--promotion--qwen3-8b-aaai27-flagship-inpo-avg-s42/snapshots/2ff371a07608f546211176b2ec6460126f2c41ea
HT_HELP=$CACHE/models--promotion--qwen3-8b-aaai27-flagship-ht-mnpo-helpfulness-s42/snapshots/057b3dce0d42d9d97d3ebf4206dbed67c672d089
HT_SAFE=$CACHE/models--promotion--qwen3-8b-aaai27-flagship-ht-mnpo-safety-s42/snapshots/97e3d067e73e2bb3da8eefe243ef4a7052318385
HT_CONCISE=$CACHE/models--promotion--qwen3-8b-aaai27-flagship-ht-mnpo-conciseness-s42/snapshots/e26d1c0940505c3869f21a168911ce49669a973a

run_one() {
  local gpu=$1 name=$2 model=$3 probe=$4
  while [[ ! -f "$model/config.json" ]]; do
    printf '%s waiting_for_model %s %s\n' "$(date -Is)" "$name" "$model" >>"$EXP/logs/decode/worker_gpu${gpu}.log"
    sleep 30
  done
  printf '%s start %s gpu=%s\n' "$(date -Is)" "$name" "$gpu" >>"$EXP/logs/decode/worker_gpu${gpu}.log"
  mkdir -p "$EXP/generations/$name"
  CUDA_VISIBLE_DEVICES=$gpu python "$DECODER" \
    --manifest "$MANIFEST" --model "$model" --policy-name "$name" --probe "$probe" \
    --output "$EXP/generations/$name/output_42.json" \
    --seed 42 --temperature 0.7 --top-p 0.9 --max-new-tokens 2048 \
    >"$EXP/logs/decode/${name}.log" 2>&1
  printf '%s complete %s gpu=%s\n' "$(date -Is)" "$name" "$gpu" >>"$EXP/logs/decode/worker_gpu${gpu}.log"
}

worker0() {
  run_one 0 base "$BASE" none
  run_one 0 ronpo_full_expect "$RONPO_FULL" none
  run_one 0 sppo_avg "$SPPO" none
  run_one 0 ht_mnpo_helpfulness "$HT_HELP" none
}
worker1() {
  run_one 1 over_refusing "$BASE" over_refusing
  run_one 1 ronpo_k_only "$RONPO_K" none
  run_one 1 inpo_avg "$INPO" none
  run_one 1 ht_mnpo_safety "$HT_SAFE" none
}
worker2() {
  run_one 2 terse "$BASE" terse
  run_one 2 ipo "$IPO" none
  run_one 2 ht_mnpo_conciseness "$HT_CONCISE" none
  run_one 2 weak_small "$WEAK" none
}
worker3() {
  run_one 3 answer_anything "$BASE" answer_anything
  run_one 3 simpo "$SIMPO" none
}

worker0 & p0=$!
worker1 & p1=$!
worker2 & p2=$!
worker3 & p3=$!
wait "$p0" "$p1" "$p2" "$p3"
date -Is >"$EXP/generations/DECODE_COMPLETE"
