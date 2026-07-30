#!/usr/bin/env bash
# Decode, reward-score, and aggregate the two new Stage-1 models on exactly
# the pre-existing P4 validation panel.  No model selection or sealed prompt
# access occurs in this runner.
set -euo pipefail

PROJECT=${PROJECT:-/NHNHOME/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/AIPR/sjkim/venv_clean}
ROOT=${ROOT:-/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1}
EXP=${EXP:-$PROJECT/results/p5_8b_robust_stage1_stage2_20260717}
P4=${P4:-$PROJECT/results/p4_8b_saferlhf_table4_20260717}
BASE=$ROOT/base_objective_screen/hf_ipv4/llama31
REWARD=$ROOT/flagship_20260712/cache/huggingface/models--PKU-Alignment--beaver-7b-v1.0-reward/snapshots/375cd6a9f0d7e339d2199b05ba129a4a8906596d
COST=$ROOT/flagship_20260712/cache/huggingface/models--PKU-Alignment--beaver-7b-v1.0-cost/snapshots/c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28
MANIFEST=$P4/dataset_manifest/validation_conflict.jsonl
OUT=$EXP/stage1/fixed_p4_validation
EXPECTED=$(wc -l < "$MANIFEST")
MODELS=(base ronpo_os_confirmatory inpo_avg sppo_avg simpo ipo dpo ht_mnpo_harmless ht_mnpo_helpfulness ronpo_topmass_stage1_replicate ronpo_softmin_lb_stage1)
NEW=(ronpo_topmass_stage1_replicate ronpo_softmin_lb_stage1)

source "$VENV/bin/activate"
export PYTHONPATH="$PROJECT" VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
export TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1
mkdir -p "$OUT"/{generations,gates,logs,shards,score_shards,scores}

for name in base ronpo_os_confirmatory inpo_avg sppo_avg simpo ipo dpo ht_mnpo_harmless ht_mnpo_helpfulness; do
  ln -sfn "$P4/validation/generations/$name" "$OUT/generations/$name"
done

decode_one() {
  local gpu=$1 name=$2 model=$3 out="$OUT/generations/$name/output_42.json"
  mkdir -p "$(dirname "$out")"
  if [[ -s "$out" ]] && "$VENV/bin/python" - "$out" "$EXPECTED" <<'PY'
import json,sys
raise SystemExit(0 if len(json.load(open(sys.argv[1]))) == int(sys.argv[2]) else 1)
PY
  then return 0; fi
  CUDA_VISIBLE_DEVICES=$gpu "$VENV/bin/python" "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py" \
    --manifest "$MANIFEST" --model "$model" --policy-name "$name" --probe none --output "$out" \
    --seed 42 --temperature 0.7 --top-p 0.9 --max-new-tokens 512 --max-model-len 8192 --gpu-memory-utilization 0.88 \
    > "$OUT/logs/decode_${name}.log" 2>&1
}
decode_one 0 ronpo_topmass_stage1_replicate "$EXP/stage1/full/ronpo_topmass_stage1_replicate" & a=$!
decode_one 1 ronpo_softmin_lb_stage1 "$EXP/stage1/full/ronpo_softmin_lb_stage1" & b=$!
# `wait pid1 pid2` reports only pid2's status.  Wait separately so a failed
# decode cannot be mistaken for a successful two-model evaluation.
wait "$a"
wait "$b"

for name in "${MODELS[@]}"; do
  "$VENV/bin/python" "$PROJECT/scripts/revision/flagship/stability_gate_corrected.py" \
    --base "$OUT/generations/base/output_42.json" --candidate "$OUT/generations/$name/output_42.json" \
    --output "$OUT/gates/$name.json" --expected-records "$EXPECTED" --min-length-ratio 0.33 --max-length-ratio 2.0 --max-repeat-run 20 \
    > "$OUT/logs/gate_${name}.log" 2>&1 || true
done
for name in "${NEW[@]}"; do
  "$VENV/bin/python" - "$OUT/gates/$name.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); assert d["passed"], d
PY
done
MODEL_CSV=$(IFS=,; echo "${MODELS[*]}")
"$VENV/bin/python" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/merge_eval_pool.py" \
  --generation-root "$OUT/generations" --models "$MODEL_CSV" --seed 42 --expected-records "$EXPECTED" --gate-root "$OUT/gates" \
  --output "$OUT/response_pool.jsonl" --audit "$OUT/pool_audit.json" > "$OUT/logs/merge_eval_pool.log" 2>&1
"$VENV/bin/python" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py" split \
  --input "$OUT/response_pool.jsonl" --output-dir "$OUT/shards" --num-shards 2 --expected-records "$EXPECTED" > "$OUT/logs/shard_score_input.log" 2>&1
score_reward() { CUDA_VISIBLE_DEVICES=$1 "$VENV/bin/python" "$PROJECT/on_policy_data_gen/rm_beaver_reward.py" --model_name "$REWARD" --model_revision 375cd6a9f0d7e339d2199b05ba129a4a8906596d --input_file "$OUT/shards/shard_$2.jsonl" --output_file "$OUT/score_shards/helpfulness_$2.jsonl" --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 > "$OUT/logs/helpfulness_$2.log" 2>&1; }
score_cost() { CUDA_VISIBLE_DEVICES=$1 "$VENV/bin/python" "$PROJECT/on_policy_data_gen/rm_beaver_cost.py" --model_name "$COST" --model_revision c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28 --input_file "$OUT/shards/shard_$2.jsonl" --output_file "$OUT/score_shards/harmlessness_$2.jsonl" --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 > "$OUT/logs/harmlessness_$2.log" 2>&1; }
# Each reward head must score both prompt shards.  Pair complementary heads
# across the two GPUs, then wait for every child explicitly before merging.
score_reward 0 0 & a=$!
score_cost 1 1 & b=$!
wait "$a"
wait "$b"
score_cost 0 0 & a=$!
score_reward 1 1 & b=$!
wait "$a"
wait "$b"
SCORES=$("$VENV/bin/python" - "$OUT/pool_audit.json" <<'PY'
import json,sys
print(len(json.load(open(sys.argv[1]))["eligible_models"]))
PY
)
"$VENV/bin/python" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py" merge --inputs "$OUT/score_shards/helpfulness_0.jsonl" "$OUT/score_shards/helpfulness_1.jsonl" --output "$OUT/scores/helpfulness.jsonl" --audit "$OUT/scores/helpfulness_audit.json" --expected-records "$EXPECTED" --expected-scores-per-row "$SCORES" --strip-responses > "$OUT/logs/merge_helpfulness.log" 2>&1
"$VENV/bin/python" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py" merge --inputs "$OUT/score_shards/harmlessness_0.jsonl" "$OUT/score_shards/harmlessness_1.jsonl" --output "$OUT/scores/harmlessness.jsonl" --audit "$OUT/scores/harmlessness_audit.json" --expected-records "$EXPECTED" --expected-scores-per-row "$SCORES" --strip-responses > "$OUT/logs/merge_harmlessness.log" 2>&1
"$VENV/bin/python" "$PROJECT/analysis/p5_8b_robust_stage1_stage2_20260717/aggregate_stage1_comparison.py" --helpfulness "$OUT/scores/helpfulness.jsonl" --harmlessness "$OUT/scores/harmlessness.jsonl" --pool-audit "$OUT/pool_audit.json" --output-dir "$OUT/results" --bootstrap 2000 --seed 42 > "$OUT/logs/aggregate.log" 2>&1
date -Is > "$OUT/EVAL_COMPLETE"
