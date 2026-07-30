#!/usr/bin/env bash
# Reuse Stage-4 gate generations, score each seed, and aggregate from JSONL.
set -euo pipefail
[[ $# -eq 4 ]] || { echo "usage: $0 PROJECT TRAIN_PY ROOT CACHE" >&2; exit 2; }
PROJECT=$1; TRAIN_PY=$2; ROOT=$3; CACHE=$4
REWARD=$CACHE/models--PKU-Alignment--beaver-7b-v1.0-reward/snapshots/375cd6a9f0d7e339d2199b05ba129a4a8906596d
COST=$CACHE/models--PKU-Alignment--beaver-7b-v1.0-cost/snapshots/c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28
ARMS=(ronpo_os inpo_avg sppo_avg simpo ipo dpo ht_mnpo_harmless ht_mnpo_helpfulness)
IFS=, read -r E0 E1 E2 E3 <<< "${EVAL_GPUS:-0,1,2,3}"
export PYTHONPATH=$PROJECT HF_HOME=$CACHE TOKENIZERS_PARALLELISM=false TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1
COMMON_LOCK=$ROOT/evaluation/common_eligible_arms.json
mapfile -t COMMON_ARMS < <("$TRAIN_PY" "$PROJECT/analysis/p14_qwen25_7b_stage4_seeds_20260718/compute_common_eligible_pool.py" \
  --root "$ROOT" --arms "${ARMS[@]}" --seeds 42 43 44 --output "$COMMON_LOCK")
COMMON_CSV=$(IFS=,; echo "${COMMON_ARMS[*]}")
for seed in 42 43 44; do
  OUT=$ROOT/evaluation/s$seed
  mkdir -p "$OUT"/{generations,gates,logs,shards,score_shards,scores,results}
  ln -sfn "$ROOT/shared/eval/base" "$OUT/generations/base"
  ln -sfn "$ROOT/shared/eval/base/gate.json" "$OUT/gates/base.json"
  for arm in "${ARMS[@]}"; do
    rm -f "$OUT/generations/$arm" "$OUT/gates/$arm.json"
    source_gen=$ROOT/seeds/s$seed/stage4/generations/$arm
    source_gate=$ROOT/seeds/s$seed/stage4/gates/$arm.json
    if [[ ",${COMMON_CSV}," == *",${arm},"* ]]; then
      ln -sfn "$source_gen" "$OUT/generations/$arm"
      ln -sfn "$source_gate" "$OUT/gates/$arm.json"
    else
      "$TRAIN_PY" - "$OUT/gates/$arm.json" "$arm" <<'PY'
import json,sys
json.dump({'status':'excluded_from_common_three_seed_pool','passed':False,'arm':sys.argv[2]},open(sys.argv[1],'w'),indent=2)
PY
    fi
  done
  MODEL_CSV=base,$COMMON_CSV
  "$TRAIN_PY" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/merge_eval_pool.py" \
    --generation-root "$OUT/generations" --models "$MODEL_CSV" --seed 42 --expected-records 1000 \
    --gate-root "$OUT/gates" --output "$OUT/response_pool.jsonl" --audit "$OUT/pool_audit.json" > "$OUT/logs/merge.log" 2>&1
  "$TRAIN_PY" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py" split --input "$OUT/response_pool.jsonl" --output-dir "$OUT/shards" --num-shards 2 --expected-records 1000
  CUDA_VISIBLE_DEVICES=$E0 "$TRAIN_PY" "$PROJECT/on_policy_data_gen/rm_beaver_reward.py" --model_name "$REWARD" --model_revision 375cd6a9f0d7e339d2199b05ba129a4a8906596d --input_file "$OUT/shards/shard_0.jsonl" --output_file "$OUT/score_shards/helpfulness_0.jsonl" --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 > "$OUT/logs/helpfulness_0.log" 2>&1 & p0=$!
  CUDA_VISIBLE_DEVICES=$E1 "$TRAIN_PY" "$PROJECT/on_policy_data_gen/rm_beaver_reward.py" --model_name "$REWARD" --model_revision 375cd6a9f0d7e339d2199b05ba129a4a8906596d --input_file "$OUT/shards/shard_1.jsonl" --output_file "$OUT/score_shards/helpfulness_1.jsonl" --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 > "$OUT/logs/helpfulness_1.log" 2>&1 & p1=$!
  CUDA_VISIBLE_DEVICES=$E2 "$TRAIN_PY" "$PROJECT/on_policy_data_gen/rm_beaver_cost.py" --model_name "$COST" --model_revision c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28 --input_file "$OUT/shards/shard_0.jsonl" --output_file "$OUT/score_shards/harmlessness_0.jsonl" --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 > "$OUT/logs/harmlessness_0.log" 2>&1 & p2=$!
  CUDA_VISIBLE_DEVICES=$E3 "$TRAIN_PY" "$PROJECT/on_policy_data_gen/rm_beaver_cost.py" --model_name "$COST" --model_revision c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28 --input_file "$OUT/shards/shard_1.jsonl" --output_file "$OUT/score_shards/harmlessness_1.jsonl" --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 > "$OUT/logs/harmlessness_1.log" 2>&1 & p3=$!
  wait "$p0" "$p1" "$p2" "$p3"
  COUNT=$($TRAIN_PY - "$OUT/pool_audit.json" <<'PY'
import json,sys
print(len(json.load(open(sys.argv[1]))['eligible_models']))
PY
)
  for objective in helpfulness harmlessness; do
    "$TRAIN_PY" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py" merge \
      --inputs "$OUT/score_shards/${objective}_0.jsonl" "$OUT/score_shards/${objective}_1.jsonl" \
      --output "$OUT/scores/${objective}.jsonl" --audit "$OUT/scores/${objective}_audit.json" \
      --expected-records 1000 --expected-scores-per-row "$COUNT" --strip-responses
  done
  "$TRAIN_PY" "$PROJECT/analysis/p5_8b_robust_stage1_stage2_20260717/aggregate_stage1_comparison.py" \
    --helpfulness "$OUT/scores/helpfulness.jsonl" --harmlessness "$OUT/scores/harmlessness.jsonl" \
    --pool-audit "$OUT/pool_audit.json" --output-dir "$OUT/results" --bootstrap 2000 --seed 42 \
    --scope "Qwen2.5-7B Stage-4 SafeRLHF 1000-prompt panel; training seed $seed" \
    --report-title "Qwen2.5-7B SafeRLHF Stage-4 seed $seed" --ronpo-arm ronpo_os \
    --comparison-label "best eligible non-RONPO trained arm"
  "$TRAIN_PY" "$PROJECT/analysis/p14_qwen25_7b_stage4_seeds_20260718/build_table4.py" \
    --result-dir "$OUT/results" --run-root "$ROOT" --seed "$seed"
  date -Is > "$OUT/EVAL_COMPLETE"
done
"$TRAIN_PY" "$PROJECT/analysis/p14_qwen25_7b_stage4_seeds_20260718/build_three_seed_table.py" \
  --root "$ROOT" --output-dir "$ROOT/evaluation/three_seed"
date -Is > "$ROOT/evaluation/ALL_EVAL_COMPLETE"
