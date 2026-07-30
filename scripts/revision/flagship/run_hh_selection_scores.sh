#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/venv_clean}
EXP=$PROJECT/results/p1_8b_hh_selection_20260716
HFHOME=/NHNHOME/26msit001_A/BASE/AIPR/sjkim/revision_qwen3_8b/full_iter1/flagship_20260712/cache/huggingface
LEGACY=/NHNHOME/26msit001_A/BASE/AIPR/sjkim/baseline_repair_1p5b_20260714/cache/reward_models
INPUT=$EXP/response_pool.jsonl
mkdir -p "$EXP/scores" "$EXP/logs/scores" "$EXP/score_status"
source "$VENV/bin/activate"
export TORCH_CUDNN_SDPA_ENABLED=0

complete() { [[ -f "$1" && $(wc -l < "$1") -eq 768 ]]; }
run_logged() {
  local name=$1 output=$2; shift 2
  if complete "$output"; then
    printf '%s\tcomplete_cached\n' "$name" >"$EXP/score_status/$name.status"
    return 0
  fi
  printf '%s\t%s\trunning\n' "$(date -Is)" "$name" >"$EXP/score_status/$name.status"
  if "$@" >"$EXP/logs/scores/$name.log" 2>&1; then
    if ! complete "$output"; then
      printf '%s\t%s\tfailed_incomplete_output\n' "$(date -Is)" "$name" >"$EXP/score_status/$name.status"
      return 1
    fi
    printf '%s\t%s\tcomplete\n' "$(date -Is)" "$name" >"$EXP/score_status/$name.status"
  else
    local code=$?
    printf '%s\t%s\tfailed_exit_%s\n' "$(date -Is)" "$name" "$code" >"$EXP/score_status/$name.status"
    return "$code"
  fi
}

skywork() {
  local gpu=$1 name=$2 model=$3 revision=$4
  run_logged "$name" "$EXP/scores/$name.jsonl" env CUDA_VISIBLE_DEVICES=$gpu python "$PROJECT/on_policy_data_gen/rm_skywork.py" \
    --model_name "$model" --revision "$revision" --local_files_only \
    --input_file "$INPUT" --output_file "$EXP/scores/$name.jsonl" \
    --max_seq_length 4096 --attn_implementation eager --batch_size 16 --sample_batch_size 8
}
athene() {
  local gpu=$1
  local revision=cdf428f7b52a323b6cf4e9803e5bcba9f1fb5a59
  local model=$LEGACY/models--Nexusflow--Athene-RM-8B/snapshots/$revision
  run_logged athene "$EXP/scores/athene.jsonl" env CUDA_VISIBLE_DEVICES=$gpu python "$PROJECT/on_policy_data_gen/rm_athene.py" \
    --model_name "$model" --revision "$revision" --local_files_only \
    --input_file "$INPUT" --output_file "$EXP/scores/athene.jsonl" \
    --device cuda --batch_size 16 --sample_batch_size 8
}
armo_help() {
  local gpu=$1 revision=eb2676d20da2f2d41082289d23c59b9f7427f955
  run_logged armo_helpfulness "$EXP/scores/armo_helpfulness.jsonl" env CUDA_VISIBLE_DEVICES=$gpu python "$PROJECT/on_policy_data_gen/rm_armo.py" \
    --cache_dir "$HFHOME/hub" --revision "$revision" --local_files_only true \
    --input_file "$INPUT" --output_file "$EXP/scores/armo_helpfulness.jsonl" \
    --device cuda --batch_size 16 --sample_batch_size 8 --max_seq_length 4096 \
    --reward_attribute_name ultrafeedback-helpfulness
}
beaver() {
  local gpu=$1 name=$2 revision=$3
  local version=${name#beaver_}
  local model=$HFHOME/models--PKU-Alignment--beaver-7b-${version}.0-cost/snapshots/$revision
  run_logged "$name" "$EXP/scores/$name.jsonl" env CUDA_VISIBLE_DEVICES=$gpu python "$PROJECT/on_policy_data_gen/rm_beaver_cost.py" \
    --model_name "$model" --model_revision "$revision" --input_file "$INPUT" \
    --output_file "$EXP/scores/$name.jsonl" --batch_size 16 --sample_batch_size 8 --max_seq_length 4096
}
guard() {
  local gpu=$1 name=$2 family=$3 model=$4 revision=$5
  run_logged "$name" "$EXP/scores/$name.jsonl" env CUDA_VISIBLE_DEVICES=$gpu python "$PROJECT/on_policy_data_gen/rm_guard_logodds.py" \
    --model_name "$model" --family "$family" --model_revision "$revision" --input_file "$INPUT" \
    --output_file "$EXP/scores/$name.jsonl" --batch_size 8 --sample_batch_size 4 --max_seq_length 4096
}

(
  skywork 0 skywork_llama "$LEGACY/models--Skywork--Skywork-Reward-V2-Llama-3.1-8B/snapshots/cba2f842f3f1af2f1b2f0d35e794d789976390c5" cba2f842f3f1af2f1b2f0d35e794d789976390c5
  guard 0 llama_guard3 llama_guard3 "$HFHOME/models--meta-llama--Llama-Guard-3-8B/snapshots/7327bd9f6efbbe6101dc6cc4736302b3cbb6e425" 7327bd9f6efbbe6101dc6cc4736302b3cbb6e425
) & p0=$!
(
  skywork 1 skywork_qwen3 "$HFHOME/models--Skywork--Skywork-Reward-V2-Qwen3-8B/snapshots/6f19fdefb933293d4898bdb59a96f7223d998659" 6f19fdefb933293d4898bdb59a96f7223d998659
  beaver 1 beaver_v1 c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28
  beaver 1 beaver_v2 26bf7161f09fee958ae64c8b4bb70fb420f7ba39
) & p1=$!
(
  athene 2
  guard 2 shieldgemma shieldgemma "$HFHOME/models--google--shieldgemma-9b/snapshots/b8b636016df4540721a098c7aab91c97ec6ee508" b8b636016df4540721a098c7aab91c97ec6ee508
) & p2=$!
(
  armo_help 3
  guard 3 qwen3guard8 qwen3guard "$HFHOME/models--Qwen--Qwen3Guard-Gen-8B/snapshots/4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb" 4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb
) & p3=$!
wait "$p0" "$p1" "$p2" "$p3"
date -Is >"$EXP/scores/SCORE_COMPLETE"
