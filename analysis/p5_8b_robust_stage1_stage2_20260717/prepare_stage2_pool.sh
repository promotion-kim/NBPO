#!/usr/bin/env bash
# Build a stage-matched, base-plus-parent Stage-2 pool.  It writes no secrets.
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 EXPERIMENT_DIR ARM PARENT_MODEL GPU_A GPU_B" >&2
  exit 2
fi

PROJECT=${PROJECT:-/NHNHOME/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/AIPR/sjkim/venv_clean}
ROOT=${ROOT:-/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1}
P4=${P4:-$PROJECT/results/p4_8b_saferlhf_table4_20260717}
EXP=$1
ARM=$2
PARENT=$3
GPU_A=$4
GPU_B=$5
BASE=$ROOT/base_objective_screen/hf_ipv4/llama31
REWARD=$ROOT/flagship_20260712/cache/huggingface/models--PKU-Alignment--beaver-7b-v1.0-reward/snapshots/375cd6a9f0d7e339d2199b05ba129a4a8906596d
COST=$ROOT/flagship_20260712/cache/huggingface/models--PKU-Alignment--beaver-7b-v1.0-cost/snapshots/c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28
MANIFEST=$P4/dataset_manifest/train_conflict.jsonl
EXPECTED=2500
POOL=$EXP/stage2/$ARM/pool

source "$VENV/bin/activate"
export PYTHONPATH="$PROJECT"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_CUDNN_SDPA_ENABLED=0
export MNPO_DISABLE_CUDNN_SDPA=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$POOL"/{generations,logs,shards,score_shards,scores,precompute}

decode_parent() {
  local gpu=$1 seed=$2
  local output="$POOL/generations/parent_seed${seed}/output_${seed}.json"
  mkdir -p "$(dirname "$output")"
  if [[ -s "$output" ]] && "$VENV/bin/python" - "$output" "$EXPECTED" <<'PY'
import json,sys
raise SystemExit(0 if len(json.load(open(sys.argv[1]))) == int(sys.argv[2]) else 1)
PY
  then return 0; fi
  CUDA_VISIBLE_DEVICES=$gpu "$VENV/bin/python" "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py" \
    --manifest "$MANIFEST" --model "$PARENT" --policy-name "${ARM}_stage1_parent_seed${seed}" --probe none \
    --output "$output" --seed "$seed" --temperature 0.7 --top-p 0.9 --max-new-tokens 512 --max-model-len 8192 --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.88}" \
    > "$POOL/logs/decode_parent_seed${seed}.log" 2>&1
}

if [[ "$GPU_A" == "$GPU_B" ]]; then
  # A one-GPU container may prepare a later baseline serially while another
  # GPU trains a completed arm.  Never place two vLLM engines on one GPU.
  decode_parent "$GPU_A" 42
  decode_parent "$GPU_B" 43
else
  decode_parent "$GPU_A" 42 & p42=$!
  decode_parent "$GPU_B" 43 & p43=$!
  wait "$p42" "$p43"
fi

"$VENV/bin/python" "$PROJECT/analysis/p5_8b_robust_stage1_stage2_20260717/merge_policy_mixture.py" \
  --base-42 "$P4/train_pool/generations/seed44/output_44.json" \
  --base-43 "$P4/train_pool/generations/seed45/output_45.json" \
  --parent-42 "$POOL/generations/parent_seed42/output_42.json" \
  --parent-43 "$POOL/generations/parent_seed43/output_43.json" \
  --parent-name "$ARM" --expected-records "$EXPECTED" --output "$POOL/response_pool.jsonl" --audit "$POOL/pool_audit.json" \
  > "$POOL/logs/merge_policy_mixture.log" 2>&1

"$VENV/bin/python" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py" split \
  --input "$POOL/response_pool.jsonl" --output-dir "$POOL/shards" --num-shards 2 --expected-records "$EXPECTED" \
  > "$POOL/logs/shard_score_input.log" 2>&1

score_reward() { CUDA_VISIBLE_DEVICES=$1 "$VENV/bin/python" "$PROJECT/on_policy_data_gen/rm_beaver_reward.py" --model_name "$REWARD" --model_revision 375cd6a9f0d7e339d2199b05ba129a4a8906596d --input_file "$POOL/shards/shard_$2.jsonl" --output_file "$POOL/score_shards/helpfulness_$2.jsonl" --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 > "$POOL/logs/helpfulness_$2.log" 2>&1; }
score_cost() { CUDA_VISIBLE_DEVICES=$1 "$VENV/bin/python" "$PROJECT/on_policy_data_gen/rm_beaver_cost.py" --model_name "$COST" --model_revision c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28 --input_file "$POOL/shards/shard_$2.jsonl" --output_file "$POOL/score_shards/harmlessness_$2.jsonl" --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 > "$POOL/logs/harmlessness_$2.log" 2>&1; }
if [[ "$GPU_A" == "$GPU_B" ]]; then
  score_reward "$GPU_A" 0
  score_reward "$GPU_B" 1
  score_cost "$GPU_A" 0
  score_cost "$GPU_B" 1
else
  score_reward "$GPU_A" 0 & r0=$!
  score_reward "$GPU_B" 1 & r1=$!
  wait "$r0" "$r1"
  score_cost "$GPU_A" 0 & c0=$!
  score_cost "$GPU_B" 1 & c1=$!
  wait "$c0" "$c1"
fi

"$VENV/bin/python" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py" merge \
  --inputs "$POOL/score_shards/helpfulness_0.jsonl" "$POOL/score_shards/helpfulness_1.jsonl" --output "$POOL/scores/helpfulness.jsonl" --audit "$POOL/scores/helpfulness_audit.json" --expected-records "$EXPECTED" --expected-scores-per-row 4 \
  > "$POOL/logs/merge_helpfulness.log" 2>&1
"$VENV/bin/python" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py" merge \
  --inputs "$POOL/score_shards/harmlessness_0.jsonl" "$POOL/score_shards/harmlessness_1.jsonl" --output "$POOL/scores/harmlessness.jsonl" --audit "$POOL/scores/harmlessness_audit.json" --expected-records "$EXPECTED" --expected-scores-per-row 4 \
  > "$POOL/logs/merge_harmlessness.log" 2>&1
"$VENV/bin/python" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/build_shared_pairs.py" \
  --helpfulness "$POOL/scores/helpfulness.jsonl" --harmlessness "$POOL/scores/harmlessness.jsonl" --train-output "$POOL/pairs_train.jsonl" --test-output "$POOL/pairs_test.jsonl" --summary "$POOL/pair_summary.json" --pairs-per-prompt 3 --internal-test-prompts 125 --expected-prompts "$EXPECTED" --expected-responses 4 --split-salt "p5-stage2-${ARM}" \
  > "$POOL/logs/build_shared_pairs.log" 2>&1
CUDA_VISIBLE_DEVICES=$GPU_A "$VENV/bin/python" -m accelerate.commands.launch --config_file "$PROJECT/accelerate_configs/single_gpu.yaml" --num_processes=1 -m mnpo_scripts.precompute \
  --model_name_or_path "$BASE" --ref_model "$BASE" --history_paths "$PARENT" --train_dir "$POOL/pairs_train.jsonl" --test_dir "$POOL/pairs_test.jsonl" --output_dir "$POOL/precompute/logps" --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --max_length 2048 --max_prompt_length 1024 --apply_chat_template true --auto_insert_empty_system_msg false --ronpo_target_mode none --report_to none \
  > "$POOL/logs/precompute.log" 2>&1
"$VENV/bin/python" "$PROJECT/mnpo_scripts/build_os_ronpo_targets.py" --input_dir "$POOL/precompute/logps" --output_dir "$POOL/precompute/targets" --kappas 0.1 --num_proc 12 \
  > "$POOL/logs/build_targets.log" 2>&1
date -Is > "$POOL/PREPARED"
