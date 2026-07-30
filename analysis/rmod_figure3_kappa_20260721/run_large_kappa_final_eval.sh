#!/usr/bin/env bash
# Evaluate the locked kappa={0.01,0.1,0.5,1,2}, Stage-1--4 grid once.
set -euo pipefail
[[ $# -eq 3 ]] || { echo "usage: $0 PROJECT NEW_ROOT OLD_ROOT" >&2; exit 2; }
PROJECT=$1; ROOT=$2; OLD=$3; OUT=$ROOT/eval_joint; VENV=$PROJECT/../venv_clean
DECODER=$PROJECT/analysis/p11_saferlhf_stage_curve_20260718/decode_and_gate.sh
mkdir -p "$OUT/generations" "$OUT/gates" "$OUT/logs/final"

for label in k1 k2; do
  while [[ ! -s "$ROOT/$label/COMPLETE" ]]; do
    if find "$ROOT/$label" -name job_status.json -type f -exec grep -l '"status": "failed"' {} + 2>/dev/null | grep -q .; then
      echo "$label chain failed" >&2; exit 1
    fi
    sleep 30
  done
done

for key in base k0p01_stage1 k0p01_stage2 k0p01_stage3 k0p01_stage4 \
  k0p1_stage1 k0p1_stage2 k0p1_stage3 k0p1_stage4 \
  k0p5_stage1 k0p5_stage2 k0p5_stage3 k0p5_stage4; do
  mkdir -p "$OUT/generations/$key"
  ln -sfn "$OLD/eval_joint/generations/$key/output_42.json" "$OUT/generations/$key/output_42.json"
  ln -sfn "$OLD/eval_joint/gates/$key.json" "$OUT/gates/$key.json"
done

for label in k1 k2; do
  arm=ronpo_os_${label}
  for stage in stage2 stage3 stage4; do
    src=$ROOT/$label/${stage}_run/${stage}_stability_p8_locked_panel
    key=${label}_${stage}
    mkdir -p "$OUT/generations/$key"
    ln -sfn "$src/generations/$arm/output_42.json" "$OUT/generations/$key/output_42.json"
    ln -sfn "$src/gates/$arm.json" "$OUT/gates/$key.json"
  done
done

bash "$DECODER" "$PROJECT" "$OUT" k1_stage1 "$ROOT/k1/stage1/full" 0 & p1=$!
bash "$DECODER" "$PROJECT" "$OUT" k2_stage1 "$ROOT/k2/stage1/full" 1 & p2=$!
wait "$p1" "$p2"

models=()
for label in k0p01 k0p1 k0p5 k1 k2; do
  for stage in 1 2 3 4; do models+=("${label}_stage${stage}"); done
done
csv=$(IFS=,; echo "base,${models[*]}")
mkdir -p "$OUT/shards" "$OUT/score_shards" "$OUT/scores" "$OUT/results"
"$VENV/bin/python" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/merge_eval_pool.py" \
  --generation-root "$OUT/generations" --models "$csv" --seed 42 --expected-records 1000 \
  --gate-root "$OUT/gates" --output "$OUT/response_pool.jsonl" --audit "$OUT/pool_audit.json"
"$VENV/bin/python" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py" split \
  --input "$OUT/response_pool.jsonl" --output-dir "$OUT/shards" --num-shards 4 --expected-records 1000
eligible=$("$VENV/bin/python" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["eligible_models"]))' "$OUT/pool_audit.json")
EARLY=$OUT/pre_scores/early19
if [[ -s "$EARLY/COMPLETE" ]]; then
  LATE=$OUT/pre_scores/late3
  mkdir -p "$LATE"/{generations,gates,shards,score_shards,scores,logs}
  for key in base k1_stage4 k2_stage4; do
    mkdir -p "$LATE/generations/$key"
    ln -sfn "$OUT/generations/$key/output_42.json" "$LATE/generations/$key/output_42.json"
    ln -sfn "$OUT/gates/$key.json" "$LATE/gates/$key.json"
  done
  "$VENV/bin/python" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/merge_eval_pool.py" \
    --generation-root "$LATE/generations" --models base,k1_stage4,k2_stage4 --seed 42 --expected-records 1000 \
    --gate-root "$LATE/gates" --output "$LATE/response_pool.jsonl" --audit "$LATE/pool_audit.json"
  "$VENV/bin/python" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py" split \
    --input "$LATE/response_pool.jsonl" --output-dir "$LATE/shards" --num-shards 2 --expected-records 1000
  pids=()
  for gpu in 0 1; do
    bash "$PROJECT/analysis/p11_saferlhf_stage_curve_20260718/score_worker.sh" "$PROJECT" "$LATE" "$gpu" "$gpu" \
      > "$LATE/logs/score_$gpu.log" 2>&1 & pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  for objective in helpfulness harmlessness; do
    "$VENV/bin/python" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py" merge \
      --inputs "$LATE/score_shards/${objective}_0.jsonl" "$LATE/score_shards/${objective}_1.jsonl" \
      --output "$LATE/scores/$objective.jsonl" --audit "$LATE/scores/${objective}_audit.json" \
      --expected-records 1000 --expected-scores-per-row 3 --strip-responses
    "$VENV/bin/python" "$PROJECT/analysis/rmod_figure3_kappa_20260721/merge_precomputed_score_sets.py" \
      --audits "$EARLY/pool_audit.json" "$LATE/pool_audit.json" \
      --scores "$EARLY/scores/$objective.jsonl" "$LATE/scores/$objective.jsonl" \
      --final-audit "$OUT/pool_audit.json" --output "$OUT/scores/$objective.jsonl" \
      > "$OUT/logs/final/merge_${objective}.log"
  done
else
  pids=()
  for gpu in 0 1 2 3; do
    bash "$PROJECT/analysis/p11_saferlhf_stage_curve_20260718/score_worker.sh" "$PROJECT" "$OUT" "$gpu" "$gpu" \
      > "$OUT/logs/final/score_$gpu.log" 2>&1 & pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  for objective in helpfulness harmlessness; do
    "$VENV/bin/python" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py" merge \
      --inputs "$OUT/score_shards/${objective}_0.jsonl" "$OUT/score_shards/${objective}_1.jsonl" \
        "$OUT/score_shards/${objective}_2.jsonl" "$OUT/score_shards/${objective}_3.jsonl" \
      --output "$OUT/scores/$objective.jsonl" --audit "$OUT/scores/${objective}_audit.json" \
      --expected-records 1000 --expected-scores-per-row "$eligible" --strip-responses
  done
fi
"$VENV/bin/python" "$PROJECT/analysis/p5_8b_robust_stage1_stage2_20260717/aggregate_stage1_comparison.py" \
  --helpfulness "$OUT/scores/helpfulness.jsonl" --harmlessness "$OUT/scores/harmlessness.jsonl" \
  --pool-audit "$OUT/pool_audit.json" --output-dir "$OUT/results" --bootstrap 2000 --seed 42 \
  --scope "common P8 1000-prompt panel; all gate-passing locked kappa-stage points" \
  --report-title "RONPO kappa-by-stage SafeRLHF trajectories" --ronpo-arm k0p1_stage4 \
  --comparison-label "other kappa-stage points"
"$VENV/bin/python" "$PROJECT/analysis/rmod_figure3_kappa_20260721/finalize_kappa_figure.py" \
  --results "$OUT/results" --out "$OUT/saferlhf_kappa_stage_front.pdf"
date -Is > "$OUT/COMPLETED"
