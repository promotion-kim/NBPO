#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 PROJECT ROOT" >&2; exit 2; }
PROJECT=$1; ROOT=$2; OUT=$ROOT/eval_joint; VENV=$PROJECT/../venv_clean
while [[ ! -s "$OUT/EXISTING_POINTS_DECODED" ]]; do sleep 30; done
for label in k0p01 k0p5; do
  while :; do
    [[ -s "$ROOT/$label/COMPLETE" ]] && break
    if find "$ROOT/$label" -path '*/gates/*.json' -type f -exec grep -l '"status": "failed"' {} + 2>/dev/null | grep -q .; then break; fi
    sleep 30
  done
done

for label in k0p01 k0p5; do
  arm=ronpo_os_${label}
  for stage in stage2 stage3 stage4; do
    exp=$ROOT/$label/${stage}_run
    src=$exp/${stage}_stability_p8_locked_panel
    key=${label}_${stage}
    if [[ -s "$src/generations/$arm/output_42.json" && -s "$src/gates/$arm.json" ]]; then
      mkdir -p "$OUT/generations/$key"
      ln -sfn "$src/generations/$arm/output_42.json" "$OUT/generations/$key/output_42.json"
      ln -sfn "$src/gates/$arm.json" "$OUT/gates/$key.json"
    fi
  done
done

models=()
for key in base k0p01_stage1 k0p01_stage2 k0p01_stage3 k0p01_stage4 k0p1_stage1 k0p1_stage2 k0p1_stage3 k0p1_stage4 k0p5_stage1 k0p5_stage2 k0p5_stage3 k0p5_stage4; do
  [[ -s "$OUT/generations/$key/output_42.json" && -s "$OUT/gates/$key.json" ]] && models+=("$key")
done
csv=$(IFS=,; echo "${models[*]}")
mkdir -p "$OUT/shards" "$OUT/score_shards" "$OUT/scores" "$OUT/results" "$OUT/logs/final"
"$VENV/bin/python" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/merge_eval_pool.py" \
  --generation-root "$OUT/generations" --models "$csv" --seed 42 --expected-records 1000 \
  --gate-root "$OUT/gates" --output "$OUT/response_pool.jsonl" --audit "$OUT/pool_audit.json"
"$VENV/bin/python" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py" split \
  --input "$OUT/response_pool.jsonl" --output-dir "$OUT/shards" --num-shards 4 --expected-records 1000

while :; do
  idle=1
  for gpu in 0 1 2 3; do
    [[ -z $(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]') ]] || idle=0
  done
  [[ $idle -eq 1 ]] && break
  sleep 20
done
pids=()
for gpu in 0 1 2 3; do
  bash "$PROJECT/analysis/p11_saferlhf_stage_curve_20260718/score_worker.sh" "$PROJECT" "$OUT" "$gpu" "$gpu" > "$OUT/logs/final/score_$gpu.log" 2>&1 & pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p"; done
eligible=$("$VENV/bin/python" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["eligible_models"]))' "$OUT/pool_audit.json")
for objective in helpfulness harmlessness; do
  "$VENV/bin/python" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py" merge \
    --inputs "$OUT/score_shards/${objective}_0.jsonl" "$OUT/score_shards/${objective}_1.jsonl" "$OUT/score_shards/${objective}_2.jsonl" "$OUT/score_shards/${objective}_3.jsonl" \
    --output "$OUT/scores/$objective.jsonl" --audit "$OUT/scores/${objective}_audit.json" \
    --expected-records 1000 --expected-scores-per-row "$eligible" --strip-responses
done
"$VENV/bin/python" "$PROJECT/analysis/p5_8b_robust_stage1_stage2_20260717/aggregate_stage1_comparison.py" \
  --helpfulness "$OUT/scores/helpfulness.jsonl" --harmlessness "$OUT/scores/harmlessness.jsonl" \
  --pool-audit "$OUT/pool_audit.json" --output-dir "$OUT/results" --bootstrap 2000 --seed 42 \
  --scope "common P8 1000-prompt panel; all gate-passing kappa-stage points" \
  --report-title "RONPO kappa-by-stage SafeRLHF trajectories" --ronpo-arm k0p1_stage4 \
  --comparison-label "other kappa-stage points"
"$VENV/bin/python" "$PROJECT/analysis/rmod_figure3_kappa_20260721/finalize_kappa_figure.py" \
  --results "$OUT/results" --out "$OUT/saferlhf_kappa_stage_front.pdf"
date -Is > "$OUT/COMPLETED"
