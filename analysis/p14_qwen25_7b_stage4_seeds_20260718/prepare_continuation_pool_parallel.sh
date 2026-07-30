#!/usr/bin/env bash
# Build one continuation pool with decode and reward scoring sharded over four GPUs.
set -euo pipefail
[[ $# -eq 8 ]] || { echo "usage: $0 PROJECT PY ROOT CACHE SEED STAGE ARM GPUS" >&2; exit 2; }
PROJECT=$1; PY=$2; ROOT=$3; CACHE=$4; SEED=$5; STAGE=$6; ARM=$7
TRAIN_PY=${TRAIN_PY:-$PY}
IFS=, read -r G0 G1 G2 G3 <<< "$8"
BASE=$CACHE/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
REWARD=$CACHE/models--PKU-Alignment--beaver-7b-v1.0-reward/snapshots/375cd6a9f0d7e339d2199b05ba129a4a8906596d
COST=$CACHE/models--PKU-Alignment--beaver-7b-v1.0-cost/snapshots/c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28
MANIFEST=$ROOT/dataset_manifest/train_conflict.jsonl
PARENT=$ROOT/seeds/s$SEED/stage$((STAGE-1))/$ARM/train/full
POOL=$ROOT/seeds/s$SEED/stage$STAGE/$ARM/pool
EXPECTED=2500
[[ -s $ROOT/run_lock.json && -s $ROOT/kappa_lock.json && -f $PARENT/config.json ]] || exit 1
mkdir -p "$POOL"/{generations,logs,shards,score_shards,scores,precompute}
export PYTHONPATH=$PROJECT HF_HOME=$CACHE VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
export TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1 MNPO_DISABLE_APEX=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

decode() {
  local sample_seed=$1 gpu=$2 out
  out=$POOL/generations/parent_seed${sample_seed}/output_${sample_seed}.json
  mkdir -p "$(dirname "$out")"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py" \
    --manifest "$MANIFEST" --model "$PARENT" --policy-name "q25_s${SEED}_stage$((STAGE-1))_${ARM}_seed${sample_seed}" \
    --probe none --output "$out" --seed "$sample_seed" --temperature 0.7 --top-p 0.9 --max-new-tokens 512 \
    --max-model-len 8192 --gpu-memory-utilization 0.88 > "$POOL/logs/decode_parent_seed${sample_seed}.log" 2>&1
}
decode 42 "$G0" & p0=$!; decode 43 "$G1" & p1=$!; wait "$p0" "$p1"

"$PY" "$PROJECT/analysis/p5_8b_robust_stage1_stage2_20260717/merge_policy_mixture.py" \
  --base-42 "$ROOT/shared/stage1_pool/generations/seed44/output_44.json" \
  --base-43 "$ROOT/shared/stage1_pool/generations/seed45/output_45.json" \
  --parent-42 "$POOL/generations/parent_seed42/output_42.json" --parent-43 "$POOL/generations/parent_seed43/output_43.json" \
  --parent-name "$ARM" --expected-records "$EXPECTED" --output "$POOL/response_pool.jsonl" --audit "$POOL/pool_audit.json"
"$PY" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py" split \
  --input "$POOL/response_pool.jsonl" --output-dir "$POOL/shards" --num-shards 2 --expected-records "$EXPECTED"

score() {
  local objective=$1 shard=$2 gpu=$3 model script revision
  if [[ $objective == helpfulness ]]; then model=$REWARD; script=rm_beaver_reward.py; revision=375cd6a9f0d7e339d2199b05ba129a4a8906596d
  else model=$COST; script=rm_beaver_cost.py; revision=c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28; fi
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$PROJECT/on_policy_data_gen/$script" --model_name "$model" --model_revision "$revision" \
    --input_file "$POOL/shards/shard_$shard.jsonl" --output_file "$POOL/score_shards/${objective}_$shard.jsonl" \
    --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 > "$POOL/logs/${objective}_$shard.log" 2>&1
}
score helpfulness 0 "$G0" & p0=$!; score helpfulness 1 "$G1" & p1=$!
score harmlessness 0 "$G2" & p2=$!; score harmlessness 1 "$G3" & p3=$!; wait "$p0" "$p1" "$p2" "$p3"

for objective in helpfulness harmlessness; do
  "$PY" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py" merge \
    --inputs "$POOL/score_shards/${objective}_0.jsonl" "$POOL/score_shards/${objective}_1.jsonl" \
    --output "$POOL/scores/${objective}.jsonl" --audit "$POOL/scores/${objective}_audit.json" \
    --expected-records "$EXPECTED" --expected-scores-per-row 4
done
"$PY" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/build_shared_pairs.py" \
  --helpfulness "$POOL/scores/helpfulness.jsonl" --harmlessness "$POOL/scores/harmlessness.jsonl" \
  --train-output "$POOL/pairs_train.jsonl" --test-output "$POOL/pairs_test.jsonl" --summary "$POOL/pair_summary.json" \
  --pairs-per-prompt 3 --internal-test-prompts 125 --expected-prompts "$EXPECTED" --expected-responses 4 \
  --split-salt "q25-s${SEED}-stage${STAGE}-${ARM}"
CUDA_VISIBLE_DEVICES=$G0 "$TRAIN_PY" -m accelerate.commands.launch --config_file "$PROJECT/accelerate_configs/single_gpu.yaml" \
  --num_processes=1 -m mnpo_scripts.precompute --model_name_or_path "$BASE" --ref_model "$BASE" --history_paths "$PARENT" \
  --train_dir "$POOL/pairs_train.jsonl" --test_dir "$POOL/pairs_test.jsonl" --output_dir "$POOL/precompute/logps" \
  --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --max_length 2048 --max_prompt_length 1024 \
  --apply_chat_template true --auto_insert_empty_system_msg false --ronpo_target_mode none --report_to none \
  > "$POOL/logs/precompute.log" 2>&1
KAPPA=$("$TRAIN_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["confirmatory_os_kappa"])' "$ROOT/kappa_lock.json")
"$TRAIN_PY" "$PROJECT/mnpo_scripts/build_os_ronpo_targets.py" --input_dir "$POOL/precompute/logps" \
  --output_dir "$POOL/precompute/targets" --kappas "$KAPPA" --num_proc 12 > "$POOL/logs/build_targets.log" 2>&1
date -Is > "$POOL/PREPARED"
