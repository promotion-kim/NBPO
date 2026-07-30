#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/venv_clean}
EXP=$PROJECT/results/p2_8b_hh_multiobjective_20260717
ANALYSIS=$PROJECT/analysis/p2_8b_hh_multiobjective_20260717
ROOT=/NHNHOME/26msit001_A/BASE/AIPR/sjkim/revision_qwen3_8b/full_iter1
BASE=$ROOT/base_objective_screen/hf_ipv4/llama31
WILD=$(tail -1 "$PROJECT/results/p1_8b_base_objective_screen_20260716/downloads/wildguard.path" | sed -n 's/^  path: //p; t; p')
GUARD=$ROOT/flagship_20260712/cache/huggingface/models--Qwen--Qwen3Guard-Gen-8B/snapshots/4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb
MANIFEST=$EXP/dataset_manifest/train.jsonl
POOL=$EXP/train_pool

mkdir -p "$POOL/generations" "$POOL/logs" "$POOL/shards" "$POOL/scores" "$POOL/precompute"
source "$VENV/bin/activate"
export PYTHONPATH=$PROJECT
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_CUDNN_SDPA_ENABLED=0
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

decode_one() {
  local gpu=$1 seed=$2
  local name=seed${seed}
  mkdir -p "$POOL/generations/$name"
  CUDA_VISIBLE_DEVICES=$gpu python "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py" \
    --manifest "$MANIFEST" --model "$BASE" --policy-name "$name" --probe none \
    --output "$POOL/generations/$name/output_${seed}.json" --seed "$seed" \
    --temperature 0.7 --top-p 0.9 --max-new-tokens 1024 \
    >"$POOL/logs/decode_${name}.log" 2>&1
}

declare -a pids
for gpu in 0 1 2 3; do
  decode_one "$gpu" "$((42 + gpu))" & pids[$gpu]=$!
done
for pid in "${pids[@]}"; do wait "$pid"; done

python "$ANALYSIS/merge_seed_pool.py" \
  --generation-root "$POOL/generations" --seeds 42,43,44,45 --expected-records 770 \
  --output "$POOL/response_pool.jsonl" --diagnostics "$POOL/generation_diagnostics.json" \
  >"$POOL/logs/merge_seed_pool.log" 2>&1

python "$ANALYSIS/shard_score_input.py" split --input "$POOL/response_pool.jsonl" \
  --output-dir "$POOL/shards" --num-shards 2 --expected-records 770 \
  >"$POOL/logs/shard_score_input.log" 2>&1

score_wild() {
  local gpu=$1 shard=$2
  CUDA_VISIBLE_DEVICES=$gpu python "$PROJECT/on_policy_data_gen/rm_wildguard_compliance.py" \
    --model_name "$WILD" --model_revision "$(basename "$WILD")" \
    --input_file "$POOL/shards/shard_${shard}.jsonl" \
    --output_file "$POOL/scores/helpfulness_shard_${shard}.jsonl" \
    --batch_size 8 --sample_batch_size 2 --max_seq_length 4096 \
    >"$POOL/logs/helpfulness_shard_${shard}.log" 2>&1
}

score_guard() {
  local gpu=$1 shard=$2
  CUDA_VISIBLE_DEVICES=$gpu python "$PROJECT/on_policy_data_gen/rm_guard_logodds.py" \
    --model_name "$GUARD" --family qwen3guard --model_revision 4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb \
    --input_file "$POOL/shards/shard_${shard}.jsonl" \
    --output_file "$POOL/scores/harmlessness_shard_${shard}.jsonl" \
    --batch_size 8 --sample_batch_size 2 --max_seq_length 4096 \
    >"$POOL/logs/harmlessness_shard_${shard}.log" 2>&1
}

score_wild 0 0 & s0=$!
score_wild 1 1 & s1=$!
score_guard 2 0 & s2=$!
score_guard 3 1 & s3=$!
wait "$s0" "$s1" "$s2" "$s3"

python "$ANALYSIS/shard_score_input.py" merge \
  --inputs "$POOL/scores/helpfulness_shard_0.jsonl" "$POOL/scores/helpfulness_shard_1.jsonl" \
  --output "$POOL/scores/helpfulness.jsonl" --audit "$POOL/scores/helpfulness_audit.json" \
  --expected-records 770 --expected-scores-per-row 4 \
  >"$POOL/logs/merge_helpfulness.log" 2>&1
python "$ANALYSIS/shard_score_input.py" merge \
  --inputs "$POOL/scores/harmlessness_shard_0.jsonl" "$POOL/scores/harmlessness_shard_1.jsonl" \
  --output "$POOL/scores/harmlessness.jsonl" --audit "$POOL/scores/harmlessness_audit.json" \
  --expected-records 770 --expected-scores-per-row 4 \
  >"$POOL/logs/merge_harmlessness.log" 2>&1

python "$ANALYSIS/build_shared_pairs.py" \
  --helpfulness "$POOL/scores/helpfulness.jsonl" --harmlessness "$POOL/scores/harmlessness.jsonl" \
  --train-output "$POOL/pairs_train.jsonl" --test-output "$POOL/pairs_test.jsonl" \
  --summary "$POOL/pair_summary.json" --pairs-per-prompt 3 --internal-test-prompts 38 \
  >"$POOL/logs/build_shared_pairs.log" 2>&1

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file "$PROJECT/accelerate_configs/single_gpu.yaml" \
  "$PROJECT/mnpo_scripts/precompute.py" \
  --model_name_or_path "$BASE" --ref_model "$BASE" --history_paths "$BASE" \
  --train_dir "$POOL/pairs_train.jsonl" --test_dir "$POOL/pairs_test.jsonl" \
  --output_dir "$POOL/precompute/shared_logps" --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 --max_length 2048 --max_prompt_length 1024 \
  --apply_chat_template true --auto_insert_empty_system_msg false \
  --ronpo_target_mode none --report_to none \
  >"$POOL/logs/precompute.log" 2>&1

python "$PROJECT/mnpo_scripts/build_os_ronpo_targets.py" \
  --input_dir "$POOL/precompute/shared_logps" \
  --output_dir "$POOL/precompute/shared_targets" --kappas 0.01 --num_proc 12 \
  >"$POOL/logs/build_targets.log" 2>&1

date -Is >"$POOL/DATA_COMPLETE"
