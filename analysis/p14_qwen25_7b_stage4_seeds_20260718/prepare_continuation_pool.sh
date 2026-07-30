#!/usr/bin/env bash
# Build one arm-local on-policy continuation pool on its claimed GPU.
set -euo pipefail
[[ $# -eq 9 ]] || { echo "usage: $0 PROJECT TRAIN_PY INFER_PY ROOT CACHE SEED STAGE ARM GPU" >&2; exit 2; }
PROJECT=$1; TRAIN_PY=$2; INFER_PY=$3; ROOT=$4; CACHE=$5; SEED=$6; STAGE=$7; ARM=$8; GPU=$9
(( STAGE >= 2 && STAGE <= 4 )) || exit 2
BASE=$CACHE/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
REWARD=$CACHE/models--PKU-Alignment--beaver-7b-v1.0-reward/snapshots/375cd6a9f0d7e339d2199b05ba129a4a8906596d
COST=$CACHE/models--PKU-Alignment--beaver-7b-v1.0-cost/snapshots/c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28
MANIFEST=$ROOT/dataset_manifest/train_conflict.jsonl
PARENT=$ROOT/seeds/s$SEED/stage$((STAGE-1))/$ARM/train/full
POOL=$ROOT/seeds/s$SEED/stage$STAGE/$ARM/pool
EXPECTED=2500
[[ -s $ROOT/run_lock.json && -s $ROOT/kappa_lock.json && -f $PARENT/config.json ]] || { echo "continuation prerequisites missing" >&2; exit 1; }
mkdir -p "$POOL"/{generations,logs,shards,score_shards,scores,precompute}
export PYTHONPATH=$PROJECT HF_HOME=$CACHE VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
export TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1 MNPO_DISABLE_APEX=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
decode_parent() {
  local sample_seed=$1
  local out=$POOL/generations/parent_seed${sample_seed}/output_${sample_seed}.json
  mkdir -p "$(dirname "$out")"
  CUDA_VISIBLE_DEVICES=$GPU "$INFER_PY" "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py" \
    --manifest "$MANIFEST" --model "$PARENT" --policy-name "q25_s${SEED}_stage$((STAGE-1))_${ARM}_seed${sample_seed}" \
    --probe none --output "$out" --seed "$sample_seed" --temperature 0.7 --top-p 0.9 --max-new-tokens 512 \
    --max-model-len 8192 --gpu-memory-utilization 0.88 > "$POOL/logs/decode_parent_seed${sample_seed}.log" 2>&1
}
decode_parent 42
decode_parent 43
"$TRAIN_PY" "$PROJECT/analysis/p5_8b_robust_stage1_stage2_20260717/merge_policy_mixture.py" \
  --base-42 "$ROOT/shared/stage1_pool/generations/seed44/output_44.json" \
  --base-43 "$ROOT/shared/stage1_pool/generations/seed45/output_45.json" \
  --parent-42 "$POOL/generations/parent_seed42/output_42.json" --parent-43 "$POOL/generations/parent_seed43/output_43.json" \
  --parent-name "$ARM" --expected-records "$EXPECTED" --output "$POOL/response_pool.jsonl" --audit "$POOL/pool_audit.json"
"$TRAIN_PY" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py" split --input "$POOL/response_pool.jsonl" --output-dir "$POOL/shards" --num-shards 2 --expected-records "$EXPECTED"
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=$GPU "$TRAIN_PY" "$PROJECT/on_policy_data_gen/rm_beaver_reward.py" --model_name "$REWARD" --model_revision 375cd6a9f0d7e339d2199b05ba129a4a8906596d --input_file "$POOL/shards/shard_$shard.jsonl" --output_file "$POOL/score_shards/helpfulness_$shard.jsonl" --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 > "$POOL/logs/helpfulness_$shard.log" 2>&1
done
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=$GPU "$TRAIN_PY" "$PROJECT/on_policy_data_gen/rm_beaver_cost.py" --model_name "$COST" --model_revision c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28 --input_file "$POOL/shards/shard_$shard.jsonl" --output_file "$POOL/score_shards/harmlessness_$shard.jsonl" --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 > "$POOL/logs/harmlessness_$shard.log" 2>&1
done
"$TRAIN_PY" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py" merge --inputs "$POOL/score_shards/helpfulness_0.jsonl" "$POOL/score_shards/helpfulness_1.jsonl" --output "$POOL/scores/helpfulness.jsonl" --audit "$POOL/scores/helpfulness_audit.json" --expected-records "$EXPECTED" --expected-scores-per-row 4
"$TRAIN_PY" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py" merge --inputs "$POOL/score_shards/harmlessness_0.jsonl" "$POOL/score_shards/harmlessness_1.jsonl" --output "$POOL/scores/harmlessness.jsonl" --audit "$POOL/scores/harmlessness_audit.json" --expected-records "$EXPECTED" --expected-scores-per-row 4
"$TRAIN_PY" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/build_shared_pairs.py" \
  --helpfulness "$POOL/scores/helpfulness.jsonl" --harmlessness "$POOL/scores/harmlessness.jsonl" \
  --train-output "$POOL/pairs_train.jsonl" --test-output "$POOL/pairs_test.jsonl" --summary "$POOL/pair_summary.json" \
  --pairs-per-prompt 3 --internal-test-prompts 125 --expected-prompts "$EXPECTED" --expected-responses 4 \
  --split-salt "q25-s${SEED}-stage${STAGE}-${ARM}"
CUDA_VISIBLE_DEVICES=$GPU "$TRAIN_PY" -m accelerate.commands.launch --config_file "$PROJECT/accelerate_configs/single_gpu.yaml" --num_processes=1 -m mnpo_scripts.precompute \
  --model_name_or_path "$BASE" --ref_model "$BASE" --history_paths "$PARENT" --train_dir "$POOL/pairs_train.jsonl" \
  --test_dir "$POOL/pairs_test.jsonl" --output_dir "$POOL/precompute/logps" --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 --max_length 2048 --max_prompt_length 1024 --apply_chat_template true \
  --auto_insert_empty_system_msg false --ronpo_target_mode none --report_to none > "$POOL/logs/precompute.log" 2>&1
KAPPA=$($TRAIN_PY - "$ROOT/kappa_lock.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['confirmatory_os_kappa'])
PY
)
"$TRAIN_PY" "$PROJECT/mnpo_scripts/build_os_ronpo_targets.py" --input_dir "$POOL/precompute/logps" --output_dir "$POOL/precompute/targets" --kappas "$KAPPA" --num_proc 12 > "$POOL/logs/build_targets.log" 2>&1
date -Is > "$POOL/PREPARED"
