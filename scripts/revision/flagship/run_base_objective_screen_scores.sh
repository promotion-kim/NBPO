#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/venv_clean}
EXP=$PROJECT/results/p1_8b_base_objective_screen_20260716
HFHOME=/NHNHOME/26msit001_A/BASE/AIPR/sjkim/revision_qwen3_8b/full_iter1/flagship_20260712/cache/huggingface
LEGACY=/NHNHOME/26msit001_A/BASE/AIPR/sjkim/baseline_repair_1p5b_20260714/cache/reward_models
WILD_CACHE=/NHNHOME/26msit001_A/BASE/AIPR/sjkim/revision_qwen3_8b/full_iter1/base_objective_screen/cache
INPUT=$EXP/response_pool.jsonl
mkdir -p "$EXP/scores" "$EXP/logs/scores" "$EXP/score_status"
source "$VENV/bin/activate"
export TORCH_CUDNN_SDPA_ENABLED=0

wild_snapshot=$(tail -1 "$EXP/downloads/wildguard.path" | sed -n 's/^  path: //p; t; p')
wild_revision=$(basename "$wild_snapshot")

complete() { [[ -f "$1" && $(wc -l <"$1") -eq 640 ]]; }
run_logged() {
  local name=$1 output=$2; shift 2
  if complete "$output"; then printf '%s\t%s\tcomplete_cached\n' "$(date -Is)" "$name" >"$EXP/score_status/$name.status"; return 0; fi
  printf '%s\t%s\trunning\n' "$(date -Is)" "$name" >"$EXP/score_status/$name.status"
  if "$@" >"$EXP/logs/scores/$name.log" 2>&1; then
    complete "$output" || { printf '%s\t%s\tfailed_incomplete\n' "$(date -Is)" "$name" >"$EXP/score_status/$name.status"; return 1; }
    printf '%s\t%s\tcomplete\n' "$(date -Is)" "$name" >"$EXP/score_status/$name.status"
  else
    code=$?; printf '%s\t%s\tfailed_exit_%s\n' "$(date -Is)" "$name" "$code" >"$EXP/score_status/$name.status"; return "$code"
  fi
}

wildguard() {
  run_logged compliance "$EXP/scores/compliance.jsonl" env CUDA_VISIBLE_DEVICES=0 python "$PROJECT/on_policy_data_gen/rm_wildguard_compliance.py" \
    --model_name "$wild_snapshot" --model_revision "$wild_revision" --input_file "$INPUT" \
    --output_file "$EXP/scores/compliance.jsonl" --batch_size 8 --sample_batch_size 2 --max_seq_length 4096
}
guard() {
  local gpu=$1 name=$2 family=$3 model=$4 revision=$5
  run_logged "$name" "$EXP/scores/$name.jsonl" env CUDA_VISIBLE_DEVICES=$gpu python "$PROJECT/on_policy_data_gen/rm_guard_logodds.py" \
    --model_name "$model" --family "$family" --model_revision "$revision" --input_file "$INPUT" \
    --output_file "$EXP/scores/$name.jsonl" --batch_size 8 --sample_batch_size 2 --max_seq_length 4096
}
beaver() {
  local model=$HFHOME/models--PKU-Alignment--beaver-7b-v2.0-cost/snapshots/26bf7161f09fee958ae64c8b4bb70fb420f7ba39
  run_logged beaver_v2 "$EXP/scores/beaver_v2.jsonl" env CUDA_VISIBLE_DEVICES=1 python "$PROJECT/on_policy_data_gen/rm_beaver_cost.py" \
    --model_name "$model" --model_revision 26bf7161f09fee958ae64c8b4bb70fb420f7ba39 \
    --input_file "$INPUT" --output_file "$EXP/scores/beaver_v2.jsonl" --batch_size 16 --sample_batch_size 4 --max_seq_length 4096
}
skywork() {
  local revision=cba2f842f3f1af2f1b2f0d35e794d789976390c5
  local model=$LEGACY/models--Skywork--Skywork-Reward-V2-Llama-3.1-8B/snapshots/$revision
  run_logged skywork_quality "$EXP/scores/skywork_quality.jsonl" env CUDA_VISIBLE_DEVICES=1 python "$PROJECT/on_policy_data_gen/rm_skywork.py" \
    --model_name "$model" --revision "$revision" --local_files_only --input_file "$INPUT" \
    --output_file "$EXP/scores/skywork_quality.jsonl" --max_seq_length 4096 --attn_implementation eager --batch_size 16 --sample_batch_size 4
}

(wildguard; guard 0 llama_guard3 llama_guard3 "$HFHOME/models--meta-llama--Llama-Guard-3-8B/snapshots/7327bd9f6efbbe6101dc6cc4736302b3cbb6e425" 7327bd9f6efbbe6101dc6cc4736302b3cbb6e425) & p0=$!
(beaver; skywork) & p1=$!
guard 2 shieldgemma shieldgemma "$HFHOME/models--google--shieldgemma-9b/snapshots/b8b636016df4540721a098c7aab91c97ec6ee508" b8b636016df4540721a098c7aab91c97ec6ee508 & p2=$!
guard 3 qwen3guard8 qwen3guard "$HFHOME/models--Qwen--Qwen3Guard-Gen-8B/snapshots/4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb" 4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb & p3=$!
wait "$p0" "$p1" "$p2" "$p3"
date -Is >"$EXP/scores/SCORE_COMPLETE"
