#!/usr/bin/env bash
# Locked Beaver validation evaluation for the SafeRLHF Table-4 W1 arm set.
set -euo pipefail

PROJECT=${PROJECT:-/NHNHOME/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/AIPR/sjkim/venv_clean}
ROOT=${ROOT:-/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1}
EXP=$PROJECT/results/p4_8b_saferlhf_table4_20260717
ANALYSIS=$PROJECT/analysis/p4_8b_saferlhf_table4_20260717
MERGE=$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/merge_eval_pool.py
SHARD=$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py
MANIFEST=$EXP/dataset_manifest/validation_conflict.jsonl
BASE=$ROOT/base_objective_screen/hf_ipv4/llama31
REWARD=$ROOT/flagship_20260712/cache/huggingface/models--PKU-Alignment--beaver-7b-v1.0-reward/snapshots/375cd6a9f0d7e339d2199b05ba129a4a8906596d
COST=$ROOT/flagship_20260712/cache/huggingface/models--PKU-Alignment--beaver-7b-v1.0-cost/snapshots/c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28
REWARD_REV=375cd6a9f0d7e339d2199b05ba129a4a8906596d
COST_REV=c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28
EXPECTED=$(wc -l < "$MANIFEST")
ER=$EXP/validation
W1=(ronpo_os_confirmatory inpo_avg sppo_avg simpo ipo dpo ht_mnpo_harmless ht_mnpo_helpfulness)

source "$VENV/bin/activate"
export PYTHONPATH=$PROJECT
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_CUDNN_SDPA_ENABLED=0
export MNPO_DISABLE_CUDNN_SDPA=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$ER"/{generations,gates,logs,shards,score_shards,scores}
"$VENV/bin/python" - "$EXP/train/w1_900steps/summary.json" "$ER/training_status.json" <<'PY'
import json,sys
summary=json.load(open(sys.argv[1]))
arms={r['arm']:r for r in summary['arms']}
json.dump({'status':'complete','arms':arms},open(sys.argv[2],'w'),indent=2); print(json.dumps(arms,indent=2))
PY

# The evaluation is a new GPU launch. All three samples must be free and no
# other account's process may be present.
: > "$ER/logs/prelaunch_gpu_samples.txt"
for sample in 1 2 3; do
  date -Is >> "$ER/logs/prelaunch_gpu_samples.txt"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> "$ER/logs/prelaunch_gpu_samples.txt"
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader)
  printf '%s\n' "$apps" >> "$ER/logs/prelaunch_gpu_samples.txt"
  [[ -z "$apps" ]] || { echo "compute process present before validation evaluation; fail closed" >&2; exit 8; }
  [[ "$sample" -eq 3 ]] || sleep 2
done

decode_one() {
  # Assign each local separately: with `set -u`, expanding `$name` in the
  # same `local` statement where it is initialized is shell-version dependent.
  local gpu=$1
  local name=$2
  local model=$3
  local output="$ER/generations/$name/output_42.json"
  mkdir -p "$(dirname "$output")"
  if [[ -s "$output" ]] && "$VENV/bin/python" - "$output" "$EXPECTED" <<'PY'
import json,sys
assert len(json.load(open(sys.argv[1]))) == int(sys.argv[2])
PY
  then return 0; fi
  CUDA_VISIBLE_DEVICES=$gpu "$VENV/bin/python" "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py" \
    --manifest "$MANIFEST" --model "$model" --policy-name "$name" --probe none --output "$output" \
    --seed 42 --temperature 0.7 --top-p 0.9 --max-new-tokens 512 --max-model-len 8192 --gpu-memory-utilization 0.88 \
    > "$ER/logs/decode_${name}.log" 2>&1
}

MODELS=(base)
PATHS=("$BASE")
for arm in "${W1[@]}"; do
  status=$("$VENV/bin/python" - "$ER/training_status.json" "$arm" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['arms'].get(sys.argv[2],{}).get('status','missing'))
PY
)
  if [[ "$status" == completed ]] && [[ -f "$EXP/train/w1_900steps/$arm/checkpoint-900/config.json" ]]; then
    MODELS+=("$arm"); PATHS+=("$EXP/train/w1_900steps/$arm/checkpoint-900")
  else
    "$VENV/bin/python" - "$ER/gates/$arm.json" "$status" <<'PY'
import json,sys
json.dump({'status':'training_failed','passed':False,'training_status':sys.argv[2]},open(sys.argv[1],'w'),indent=2)
PY
  fi
done
[[ "${#MODELS[@]}" -ge 2 ]] || { echo "no trained W1 arm reached evaluation" >&2; exit 9; }
printf '%s\n' "${MODELS[@]}" | paste -sd, > "$ER/eligible_requested_models.txt"

# Decode four models at a time, one vLLM engine per authorized GPU.
pids=()
for index in "${!MODELS[@]}"; do
  gpu=$((index % 4))
  decode_one "$gpu" "${MODELS[$index]}" "${PATHS[$index]}" & pids+=("$!")
  if [[ "${#pids[@]}" -eq 4 ]]; then
    for pid in "${pids[@]}"; do wait "$pid"; done
    pids=()
  fi
done
for pid in "${pids[@]}"; do wait "$pid"; done

for name in "${MODELS[@]}"; do
  "$VENV/bin/python" "$PROJECT/scripts/revision/flagship/stability_gate_corrected.py" \
    --base "$ER/generations/base/output_42.json" --candidate "$ER/generations/$name/output_42.json" \
    --output "$ER/gates/$name.json" --expected-records "$EXPECTED" \
    --min-length-ratio 0.33 --max-length-ratio 2.0 --max-repeat-run 20 \
    > "$ER/logs/gate_${name}.log" 2>&1 || true
  [[ -s "$ER/gates/$name.json" ]]
done

MODEL_CSV=$(paste -sd, "$ER/eligible_requested_models.txt")
"$VENV/bin/python" "$MERGE" --generation-root "$ER/generations" --models "$MODEL_CSV" --seed 42 \
  --expected-records "$EXPECTED" --gate-root "$ER/gates" --output "$ER/response_pool.jsonl" --audit "$ER/pool_audit.json" \
  > "$ER/logs/merge_eval_pool.log" 2>&1
"$VENV/bin/python" "$SHARD" split --input "$ER/response_pool.jsonl" --output-dir "$ER/shards" --num-shards 2 --expected-records "$EXPECTED" \
  > "$ER/logs/shard_score_input.log" 2>&1

score_reward() { CUDA_VISIBLE_DEVICES=$1 "$VENV/bin/python" "$PROJECT/on_policy_data_gen/rm_beaver_reward.py" --model_name "$REWARD" --model_revision "$REWARD_REV" --input_file "$ER/shards/shard_$2.jsonl" --output_file "$ER/score_shards/helpfulness_$2.jsonl" --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 > "$ER/logs/helpfulness_$2.log" 2>&1; }
score_cost() { CUDA_VISIBLE_DEVICES=$1 "$VENV/bin/python" "$PROJECT/on_policy_data_gen/rm_beaver_cost.py" --model_name "$COST" --model_revision "$COST_REV" --input_file "$ER/shards/shard_$2.jsonl" --output_file "$ER/score_shards/harmlessness_$2.jsonl" --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 > "$ER/logs/harmlessness_$2.log" 2>&1; }
score_reward 0 0 & p0=$!; score_reward 1 1 & p1=$!; score_cost 2 0 & p2=$!; score_cost 3 1 & p3=$!
wait "$p0" "$p1" "$p2" "$p3"
SCORES_PER_RECORD=$("$VENV/bin/python" - "$ER/pool_audit.json" <<'PY'
import json,sys
print(len(json.load(open(sys.argv[1]))['eligible_models']))
PY
)
"$VENV/bin/python" "$SHARD" merge --inputs "$ER/score_shards/helpfulness_0.jsonl" "$ER/score_shards/helpfulness_1.jsonl" --output "$ER/scores/helpfulness.jsonl" --audit "$ER/scores/helpfulness_audit.json" --expected-records "$EXPECTED" --expected-scores-per-row "$SCORES_PER_RECORD" --strip-responses > "$ER/logs/merge_helpfulness.log" 2>&1
"$VENV/bin/python" "$SHARD" merge --inputs "$ER/score_shards/harmlessness_0.jsonl" "$ER/score_shards/harmlessness_1.jsonl" --output "$ER/scores/harmlessness.jsonl" --audit "$ER/scores/harmlessness_audit.json" --expected-records "$EXPECTED" --expected-scores-per-row "$SCORES_PER_RECORD" --strip-responses > "$ER/logs/merge_harmlessness.log" 2>&1

"$VENV/bin/python" "$ANALYSIS/build_table4_saferlhf.py" --helpfulness "$ER/scores/helpfulness.jsonl" --harmlessness "$ER/scores/harmlessness.jsonl" --pool-audit "$ER/pool_audit.json" --gate-root "$ER/gates" --calibration "$PROJECT/results/p4_8b_saferlhf_kappa_imbalance_20260717/preflight/calibration_summary.json" --data-manifest "$EXP/dataset_manifest/data_manifest.json" --train-pair-count 7125 --steps 900 --effective-batch 16 --bootstrap-resamples 2000 --seed 42 --output-dir "$EXP" > "$ER/logs/build_table4.log" 2>&1
date -Is > "$ER/EVAL_COMPLETE"
