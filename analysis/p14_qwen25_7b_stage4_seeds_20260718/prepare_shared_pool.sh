#!/usr/bin/env bash
# Build the Qwen-specific Stage-1 pool once. No training outcome is consulted.
set -euo pipefail
[[ $# -eq 5 ]] || { echo "usage: $0 PROJECT TRAIN_PY INFER_PY ROOT CACHE" >&2; exit 2; }
PROJECT=$1; TRAIN_PY=$2; INFER_PY=$3; ROOT=$4; CACHE=$5
BASE=$CACHE/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
REWARD=$CACHE/models--PKU-Alignment--beaver-7b-v1.0-reward/snapshots/375cd6a9f0d7e339d2199b05ba129a4a8906596d
COST=$CACHE/models--PKU-Alignment--beaver-7b-v1.0-cost/snapshots/c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28
MANIFEST=$ROOT/dataset_manifest/train_conflict.jsonl
EVAL_MANIFEST=$ROOT/dataset_manifest/fresh_default_test_1000.jsonl
POOL=$ROOT/shared/stage1_pool
EXPECTED=2500
[[ -s $ROOT/run_lock.json && -s $MANIFEST && -s $EVAL_MANIFEST && -d $BASE && -d $REWARD && -d $COST ]] || { echo "locked inputs or model snapshots missing" >&2; exit 1; }
mkdir -p "$POOL"/{generations,logs,shards,score_shards,scores,precompute} "$ROOT/shared/eval/base"
export PYTHONPATH=$PROJECT HF_HOME=$CACHE VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
export TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1 MNPO_DISABLE_APEX=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

decode_one() {
  local gpu=$1
  local seed=$2
  local out=$POOL/generations/seed${seed}/output_${seed}.json
  mkdir -p "$(dirname "$out")"
  CUDA_VISIBLE_DEVICES=$gpu "$INFER_PY" "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py" \
    --manifest "$MANIFEST" --model "$BASE" --policy-name "qwen25_base_seed${seed}" --probe none --output "$out" \
    --seed "$seed" --temperature 0.7 --top-p 0.9 --max-new-tokens 512 --max-model-len 8192 --gpu-memory-utilization 0.88 \
    > "$POOL/logs/decode_seed${seed}.log" 2>&1
}
if [[ ! -s $POOL/DECODE_COMPLETE ]]; then
  pids=(); for index in 0 1 2 3; do decode_one "$index" "$((42+index))" & pids+=("$!"); done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "$TRAIN_PY" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/merge_seed_pool.py" \
    --generation-root "$POOL/generations" --seeds 42,43,44,45 --expected-records "$EXPECTED" \
    --output "$POOL/response_pool.jsonl" --diagnostics "$POOL/generation_diagnostics.json"
  date -Is > "$POOL/DECODE_COMPLETE"
fi
"$TRAIN_PY" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py" split \
  --input "$POOL/response_pool.jsonl" --output-dir "$POOL/shards" --num-shards 2 --expected-records "$EXPECTED"
score_reward() { CUDA_VISIBLE_DEVICES=$1 "$TRAIN_PY" "$PROJECT/on_policy_data_gen/rm_beaver_reward.py" --model_name "$REWARD" --model_revision 375cd6a9f0d7e339d2199b05ba129a4a8906596d --input_file "$POOL/shards/shard_$2.jsonl" --output_file "$POOL/score_shards/helpfulness_$2.jsonl" --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 > "$POOL/logs/helpfulness_$2.log" 2>&1; }
score_cost() { CUDA_VISIBLE_DEVICES=$1 "$TRAIN_PY" "$PROJECT/on_policy_data_gen/rm_beaver_cost.py" --model_name "$COST" --model_revision c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28 --input_file "$POOL/shards/shard_$2.jsonl" --output_file "$POOL/score_shards/harmlessness_$2.jsonl" --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 > "$POOL/logs/harmlessness_$2.log" 2>&1; }
if [[ ! -s $POOL/SCORES_COMPLETE ]]; then
  score_reward 0 0 & p0=$!; score_reward 1 1 & p1=$!; score_cost 2 0 & p2=$!; score_cost 3 1 & p3=$!
  wait "$p0" "$p1" "$p2" "$p3"
  "$TRAIN_PY" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py" merge --inputs "$POOL/score_shards/helpfulness_0.jsonl" "$POOL/score_shards/helpfulness_1.jsonl" --output "$POOL/scores/helpfulness.jsonl" --audit "$POOL/scores/helpfulness_audit.json" --expected-records "$EXPECTED" --expected-scores-per-row 4
  "$TRAIN_PY" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py" merge --inputs "$POOL/score_shards/harmlessness_0.jsonl" "$POOL/score_shards/harmlessness_1.jsonl" --output "$POOL/scores/harmlessness.jsonl" --audit "$POOL/scores/harmlessness_audit.json" --expected-records "$EXPECTED" --expected-scores-per-row 4
  date -Is > "$POOL/SCORES_COMPLETE"
fi
"$TRAIN_PY" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/build_shared_pairs.py" \
  --helpfulness "$POOL/scores/helpfulness.jsonl" --harmlessness "$POOL/scores/harmlessness.jsonl" \
  --train-output "$POOL/pairs_train.jsonl" --test-output "$POOL/pairs_test.jsonl" --summary "$POOL/pair_summary.json" \
  --pairs-per-prompt 3 --internal-test-prompts 125 --expected-prompts "$EXPECTED" --expected-responses 4 --split-salt table4-conflict

decode_base_eval() {
  CUDA_VISIBLE_DEVICES=1 "$INFER_PY" "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py" \
    --manifest "$EVAL_MANIFEST" --model "$BASE" --policy-name qwen25_base --probe none \
    --output "$ROOT/shared/eval/base/output_42.json" --seed 42 --temperature 0.7 --top-p 0.9 \
    --max-new-tokens 512 --max-model-len 8192 --gpu-memory-utilization 0.88 \
    > "$POOL/logs/decode_base_eval.log" 2>&1
}
decode_base_eval & eval_pid=$!
CUDA_VISIBLE_DEVICES=0 "$TRAIN_PY" -m accelerate.commands.launch --config_file "$PROJECT/accelerate_configs/single_gpu.yaml" --num_processes=1 -m mnpo_scripts.precompute \
  --model_name_or_path "$BASE" --ref_model "$BASE" --history_paths "$BASE" \
  --train_dir "$POOL/pairs_train.jsonl" --test_dir "$POOL/pairs_test.jsonl" --output_dir "$POOL/precompute/logps" \
  --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --max_length 2048 --max_prompt_length 1024 \
  --apply_chat_template true --auto_insert_empty_system_msg false --ronpo_target_mode none --report_to none \
  > "$POOL/logs/precompute.log" 2>&1
wait "$eval_pid"
"$TRAIN_PY" "$PROJECT/scripts/revision/flagship/stability_gate_corrected.py" \
  --base "$ROOT/shared/eval/base/output_42.json" --candidate "$ROOT/shared/eval/base/output_42.json" \
  --output "$ROOT/shared/eval/base/gate.json" --expected-records 1000 --min-length-ratio 0.33 --max-length-ratio 2.0 --max-repeat-run 20
"$TRAIN_PY" "$PROJECT/analysis/p4_8b_saferlhf_kappa_imbalance_20260717/select_kappas_by_entropy.py" \
  --input-dir "$POOL/precompute/logps" --candidates 0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5 \
  --entropy-targets 0.55 --output "$ROOT/kappa_lock.json"
KAPPAS=$($TRAIN_PY - "$ROOT/kappa_lock.json" <<'PY'
import json,sys
print(','.join(str(x['selected_kappa']) for x in json.load(open(sys.argv[1]))['selected']))
PY
)
"$TRAIN_PY" "$PROJECT/mnpo_scripts/build_os_ronpo_targets.py" --input_dir "$POOL/precompute/logps" --output_dir "$POOL/precompute/targets" --kappas "$KAPPAS" --num_proc 12 > "$POOL/logs/build_targets.log" 2>&1
date -Is > "$POOL/PREPARED"
