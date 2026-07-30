#!/usr/bin/env bash
# Stage-matched descriptive evaluation for P5 Stage-2 models only.
# The runner accepts only models that already passed the corrected 49-record
# stability gate. It reuses those exact generated responses, never decodes a
# sealed split, and recomputes normalization over the Stage-2 eligible pool.
set -euo pipefail

PROJECT=${PROJECT:-/NHNHOME/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/AIPR/sjkim/venv_clean}
ROOT=${ROOT:-/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1}
EXP=${EXP:-$PROJECT/results/p5_8b_robust_stage1_stage2_20260717}
P4=${P4:-$PROJECT/results/p4_8b_saferlhf_table4_20260717}
REWARD=$ROOT/flagship_20260712/cache/huggingface/models--PKU-Alignment--beaver-7b-v1.0-reward/snapshots/375cd6a9f0d7e339d2199b05ba129a4a8906596d
COST=$ROOT/flagship_20260712/cache/huggingface/models--PKU-Alignment--beaver-7b-v1.0-cost/snapshots/c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28
MANIFEST=$P4/dataset_manifest/validation_conflict.jsonl
OUT=$EXP/stage2/fixed_p4_validation

usage() {
  echo "usage: $0 --model MODEL_KEY [--model MODEL_KEY ...]" >&2
  echo "MODEL_KEY must name a completed P5 Stage-2 arm with gate.json=passed." >&2
  exit 2
}
MODELS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODELS+=("$2"); shift 2 ;;
    *) usage ;;
  esac
done
[[ ${#MODELS[@]} -gt 0 ]] || usage

source "$VENV/bin/activate"
export PYTHONPATH="$PROJECT" VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
export TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1
EXPECTED=$(wc -l < "$MANIFEST")
[[ "$EXPECTED" == "49" ]] || { echo "unexpected fixed-panel size: $EXPECTED" >&2; exit 1; }
mkdir -p "$OUT"/{generations,gates,logs,shards,score_shards,scores}
ln -sfn "$P4/validation/generations/base" "$OUT/generations/base"
"$VENV/bin/python" "$PROJECT/scripts/revision/flagship/stability_gate_corrected.py" \
  --base "$OUT/generations/base/output_42.json" --candidate "$OUT/generations/base/output_42.json" \
  --output "$OUT/gates/base.json" --expected-records "$EXPECTED" --min-length-ratio 0.33 --max-length-ratio 2.0 --max-repeat-run 20 \
  > "$OUT/logs/gate_base.log" 2>&1

for name in "${MODELS[@]}"; do
  gate="$EXP/stage2/$name/stability_validation/gate.json"
  gen="$EXP/stage2/$name/stability_validation/generation"
  model="$EXP/stage2/$name/train/full"
  [[ -d "$model" && -f "$gate" && -f "$gen/output_42.json" ]] || { echo "missing Stage-2 artifact for $name" >&2; exit 1; }
  "$VENV/bin/python" - "$gate" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x.get("passed") is True, x
PY
  ln -sfn "$gen" "$OUT/generations/$name"
  "$VENV/bin/python" "$PROJECT/scripts/revision/flagship/stability_gate_corrected.py" \
    --base "$OUT/generations/base/output_42.json" --candidate "$OUT/generations/$name/output_42.json" \
    --output "$OUT/gates/$name.json" --expected-records "$EXPECTED" --min-length-ratio 0.33 --max-length-ratio 2.0 --max-repeat-run 20 \
    > "$OUT/logs/gate_${name}.log" 2>&1
  "$VENV/bin/python" - "$OUT/gates/$name.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x.get("passed") is True, x
PY
done

ALL=(base "${MODELS[@]}")
MODEL_CSV=$(IFS=,; echo "${ALL[*]}")
"$VENV/bin/python" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/merge_eval_pool.py" \
  --generation-root "$OUT/generations" --models "$MODEL_CSV" --seed 42 --expected-records "$EXPECTED" --gate-root "$OUT/gates" \
  --output "$OUT/response_pool.jsonl" --audit "$OUT/pool_audit.json" > "$OUT/logs/merge_eval_pool.log" 2>&1
"$VENV/bin/python" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py" split \
  --input "$OUT/response_pool.jsonl" --output-dir "$OUT/shards" --num-shards 2 --expected-records "$EXPECTED" \
  > "$OUT/logs/shard_score_input.log" 2>&1
score_reward() { CUDA_VISIBLE_DEVICES=$1 "$VENV/bin/python" "$PROJECT/on_policy_data_gen/rm_beaver_reward.py" --model_name "$REWARD" --model_revision 375cd6a9f0d7e339d2199b05ba129a4a8906596d --input_file "$OUT/shards/shard_$2.jsonl" --output_file "$OUT/score_shards/helpfulness_$2.jsonl" --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 > "$OUT/logs/helpfulness_$2.log" 2>&1; }
score_cost() { CUDA_VISIBLE_DEVICES=$1 "$VENV/bin/python" "$PROJECT/on_policy_data_gen/rm_beaver_cost.py" --model_name "$COST" --model_revision c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28 --input_file "$OUT/shards/shard_$2.jsonl" --output_file "$OUT/score_shards/harmlessness_$2.jsonl" --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 > "$OUT/logs/harmlessness_$2.log" 2>&1; }
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
args=(--helpfulness "$OUT/scores/helpfulness.jsonl" --harmlessness "$OUT/scores/harmlessness.jsonl" --pool-audit "$OUT/pool_audit.json" --output-dir "$OUT/results" --bootstrap 2000 --seed 42 --scope "descriptive Stage-2-only comparison on the fixed P4 49-prompt validation panel; not a fresh confirmation" --report-title "P5 Stage-2 fixed-panel comparison")
for name in "${MODELS[@]}"; do
  [[ "$name" == ronpo_* ]] && args+=(--ronpo-arm "$name")
done
"$VENV/bin/python" "$PROJECT/analysis/p5_8b_robust_stage1_stage2_20260717/aggregate_stage1_comparison.py" "${args[@]}" > "$OUT/logs/aggregate.log" 2>&1
date -Is > "$OUT/EVAL_COMPLETE"
