#!/usr/bin/env bash
# Evaluate W2 exactly once as a diagnostic.  W1 outputs are reused verbatim.
set -euo pipefail

PROJECT=/NHNHOME/AIPR/sjkim/MNPO_rev_20260710
VENV=/NHNHOME/AIPR/sjkim/venv_clean
EXP="$PROJECT/results/p4_8b_saferlhf_table4_20260717"
EVAL="$EXP/w2_validation"
BASE=/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1/base_objective_screen/hf_ipv4/llama31
REWARD=/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1/flagship_20260712/cache/huggingface/models--PKU-Alignment--beaver-7b-v1.0-reward/snapshots/375cd6a9f0d7e339d2199b05ba129a4a8906596d
COST=/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1/flagship_20260712/cache/huggingface/models--PKU-Alignment--beaver-7b-v1.0-cost/snapshots/c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28
REWARD_REV=375cd6a9f0d7e339d2199b05ba129a4a8906596d
COST_REV=c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28
MANIFEST="$EXP/dataset_manifest/validation_conflict.jsonl"
MERGE="$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/merge_eval_pool.py"
SHARD="$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py"
EXPECTED=$(wc -l < "$MANIFEST")
W1=(ronpo_os_confirmatory inpo_avg sppo_avg simpo ipo dpo ht_mnpo_harmless ht_mnpo_helpfulness)
W2=(ronpo_topmass ronpo_uniform ronpo_os_entropy_0p15_diagnostic ronpo_os_entropy_0p85_diagnostic mnpo ronpo_os_lr_1e-6_diagnostic inpo_avg_lr_1e-6_fairness ipo_lr_1e-6_fairness)

source "$VENV/bin/activate"
export PYTHONPATH="$PROJECT"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_CUDNN_SDPA_ENABLED=0
export MNPO_DISABLE_CUDNN_SDPA=1
export TOKENIZERS_PARALLELISM=false

test -f "$EXP/train/w2_900steps/summary.json"
"$VENV/bin/python" - "$EXP/train/w2_900steps/summary.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); assert d['status']=='complete', d
assert all(x['status']=='completed' for x in d['arms']), d
PY

mkdir -p "$EVAL"/{generations,gates,logs,shards,score_shards,scores}
: > "$EVAL/prelaunch_gpu_samples.txt"
for s in 1 2 3; do
  date -Is >> "$EVAL/prelaunch_gpu_samples.txt"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> "$EVAL/prelaunch_gpu_samples.txt"
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader)
  printf '%s\n' "$apps" >> "$EVAL/prelaunch_gpu_samples.txt"
  [[ -z "$apps" ]] || { echo "compute process present; fail closed" >&2; exit 8; }
  [[ "$s" -eq 3 ]] || sleep 2
done

for name in base "${W1[@]}"; do
  test -s "$EXP/validation/generations/$name/output_42.json"
  ln -sfn "$EXP/validation/generations/$name" "$EVAL/generations/$name"
done

decode_one() {
  local gpu=$1
  local name=$2
  local output="$EVAL/generations/$name/output_42.json"
  mkdir -p "$(dirname "$output")"
  if [[ -s "$output" ]] && "$VENV/bin/python" - "$output" "$EXPECTED" <<'PY'
import json,sys
assert len(json.load(open(sys.argv[1]))) == int(sys.argv[2])
PY
  then return 0; fi
  CUDA_VISIBLE_DEVICES=$gpu "$VENV/bin/python" "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py" \
    --manifest "$MANIFEST" --model "$EXP/train/w2_900steps/$name/checkpoint-900" --policy-name "$name" --probe none --output "$output" \
    --seed 42 --temperature 0.7 --top-p 0.9 --max-new-tokens 512 --max-model-len 8192 --gpu-memory-utilization 0.88 \
    > "$EVAL/logs/decode_${name}.log" 2>&1
}
pids=()
for i in "${!W2[@]}"; do
  decode_one "$((i % 4))" "${W2[$i]}" & pids+=("$!")
  if [[ "${#pids[@]}" -eq 4 ]]; then for pid in "${pids[@]}"; do wait "$pid"; done; pids=(); fi
done
for pid in "${pids[@]}"; do wait "$pid"; done

MODELS=(base "${W1[@]}" "${W2[@]}")
for name in "${MODELS[@]}"; do
  "$VENV/bin/python" "$PROJECT/scripts/revision/flagship/stability_gate_corrected.py" \
    --base "$EVAL/generations/base/output_42.json" --candidate "$EVAL/generations/$name/output_42.json" --output "$EVAL/gates/$name.json" \
    --expected-records "$EXPECTED" --min-length-ratio 0.33 --max-length-ratio 2.0 --max-repeat-run 20 > "$EVAL/logs/gate_${name}.log" 2>&1 || true
  "$VENV/bin/python" - "$EVAL/gates/$name.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); assert d['passed'], d
PY
done
MODEL_CSV=$(IFS=,; echo "${MODELS[*]}")
"$VENV/bin/python" "$MERGE" --generation-root "$EVAL/generations" --models "$MODEL_CSV" --seed 42 --expected-records "$EXPECTED" --gate-root "$EVAL/gates" --output "$EVAL/response_pool.jsonl" --audit "$EVAL/pool_audit.json" > "$EVAL/logs/merge_eval_pool.log" 2>&1
"$VENV/bin/python" "$SHARD" split --input "$EVAL/response_pool.jsonl" --output-dir "$EVAL/shards" --num-shards 2 --expected-records "$EXPECTED" > "$EVAL/logs/shard_score_input.log" 2>&1
score_reward() { CUDA_VISIBLE_DEVICES=$1 "$VENV/bin/python" "$PROJECT/on_policy_data_gen/rm_beaver_reward.py" --model_name "$REWARD" --model_revision "$REWARD_REV" --input_file "$EVAL/shards/shard_$2.jsonl" --output_file "$EVAL/score_shards/helpfulness_$2.jsonl" --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 > "$EVAL/logs/helpfulness_$2.log" 2>&1; }
score_cost() { CUDA_VISIBLE_DEVICES=$1 "$VENV/bin/python" "$PROJECT/on_policy_data_gen/rm_beaver_cost.py" --model_name "$COST" --model_revision "$COST_REV" --input_file "$EVAL/shards/shard_$2.jsonl" --output_file "$EVAL/score_shards/harmlessness_$2.jsonl" --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 > "$EVAL/logs/harmlessness_$2.log" 2>&1; }
score_reward 0 0 & a=$!; score_reward 1 1 & b=$!; score_cost 2 0 & c=$!; score_cost 3 1 & d=$!; wait "$a" "$b" "$c" "$d"
SCORES=$(( ${#MODELS[@]} ))
"$VENV/bin/python" "$SHARD" merge --inputs "$EVAL/score_shards/helpfulness_0.jsonl" "$EVAL/score_shards/helpfulness_1.jsonl" --output "$EVAL/scores/helpfulness.jsonl" --audit "$EVAL/scores/helpfulness_audit.json" --expected-records "$EXPECTED" --expected-scores-per-row "$SCORES" --strip-responses > "$EVAL/logs/merge_helpfulness.log" 2>&1
"$VENV/bin/python" "$SHARD" merge --inputs "$EVAL/score_shards/harmlessness_0.jsonl" "$EVAL/score_shards/harmlessness_1.jsonl" --output "$EVAL/scores/harmlessness.jsonl" --audit "$EVAL/scores/harmlessness_audit.json" --expected-records "$EXPECTED" --expected-scores-per-row "$SCORES" --strip-responses > "$EVAL/logs/merge_harmlessness.log" 2>&1
"$VENV/bin/python" "$PROJECT/analysis/p4_8b_saferlhf_table4_20260717/build_w2_ablation.py" --helpfulness "$EVAL/scores/helpfulness.jsonl" --harmlessness "$EVAL/scores/harmlessness.jsonl" --pool-audit "$EVAL/pool_audit.json" --output-dir "$EXP" --bootstrap 2000 --seed 42 > "$EVAL/logs/build_w2_ablation.log" 2>&1
date -Is > "$EVAL/EVAL_COMPLETE"
