#!/usr/bin/env bash
set -euo pipefail
PROJECT=${PROJECT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/venv_clean}
EXP=$PROJECT/results/p1_8b_hh_selection_20260716
CACHE=/NHNHOME/26msit001_A/BASE/AIPR/sjkim/revision_qwen3_8b/full_iter1/flagship_20260712/cache/huggingface
INPUT=$EXP/scorer_smoke_input.jsonl
mkdir -p "$EXP/scorer_smoke"
source "$VENV/bin/activate"
export TORCH_CUDNN_SDPA_ENABLED=0

beaver() {
  local gpu=$1 name=$2 revision=$3
  local model=$CACHE/models--PKU-Alignment--beaver-7b-${name#beaver_}.0-cost/snapshots/$revision
  CUDA_VISIBLE_DEVICES=$gpu python "$PROJECT/on_policy_data_gen/rm_beaver_cost.py" \
    --model_name "$model" --input_file "$INPUT" --output_file "$EXP/scorer_smoke/$name.jsonl" \
    --model_revision "$revision" --batch_size 2 --sample_batch_size 1 \
    >"$EXP/scorer_smoke/$name.log" 2>&1
}
guard() {
  local gpu=$1 name=$2 family=$3 model=$4 revision=$5
  CUDA_VISIBLE_DEVICES=$gpu python "$PROJECT/on_policy_data_gen/rm_guard_logodds.py" \
    --model_name "$model" --family "$family" --input_file "$INPUT" \
    --output_file "$EXP/scorer_smoke/$name.jsonl" --model_revision "$revision" \
    --batch_size 2 --sample_batch_size 1 \
    >"$EXP/scorer_smoke/$name.log" 2>&1
}

(
  beaver 0 beaver_v1 c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28
  beaver 0 beaver_v2 26bf7161f09fee958ae64c8b4bb70fb420f7ba39
) & p0=$!
guard 1 llama_guard3 llama_guard3 "$CACHE/models--meta-llama--Llama-Guard-3-8B/snapshots/7327bd9f6efbbe6101dc6cc4736302b3cbb6e425" 7327bd9f6efbbe6101dc6cc4736302b3cbb6e425 & p1=$!
(
  guard 3 shieldgemma shieldgemma "$CACHE/models--google--shieldgemma-9b/snapshots/b8b636016df4540721a098c7aab91c97ec6ee508" b8b636016df4540721a098c7aab91c97ec6ee508
  guard 3 qwen3guard8 qwen3guard "$CACHE/models--Qwen--Qwen3Guard-Gen-8B/snapshots/4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb" 4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb
) & p3=$!
wait "$p0" "$p1" "$p3"
date -Is >"$EXP/scorer_smoke/COMPLETE"
