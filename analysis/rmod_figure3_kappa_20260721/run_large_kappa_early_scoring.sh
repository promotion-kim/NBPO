#!/usr/bin/env bash
# Pre-score all Figure 3 points that are complete before large-kappa Stage 4.
set -euo pipefail
[[ $# -eq 3 ]] || { echo "usage: $0 PROJECT NEW_ROOT OLD_ROOT" >&2; exit 2; }
PROJECT=$1; ROOT=$2; OLD=$3; OUT=$ROOT/eval_joint/pre_scores/early19
VENV=$PROJECT/../venv_clean
mkdir -p "$OUT"/{generations,gates,shards,score_shards,scores,logs}

for key in base k0p01_stage1 k0p01_stage2 k0p01_stage3 k0p01_stage4 \
  k0p1_stage1 k0p1_stage2 k0p1_stage3 k0p1_stage4 \
  k0p5_stage1 k0p5_stage2 k0p5_stage3 k0p5_stage4; do
  mkdir -p "$OUT/generations/$key"
  ln -sfn "$OLD/eval_joint/generations/$key/output_42.json" "$OUT/generations/$key/output_42.json"
  ln -sfn "$OLD/eval_joint/gates/$key.json" "$OUT/gates/$key.json"
done
for label in k1 k2; do
  mkdir -p "$OUT/generations/${label}_stage1"
  ln -sfn "$ROOT/eval_joint/generations/${label}_stage1/output_42.json" "$OUT/generations/${label}_stage1/output_42.json"
  ln -sfn "$ROOT/eval_joint/gates/${label}_stage1.json" "$OUT/gates/${label}_stage1.json"
  for stage in stage2 stage3; do
    src=$ROOT/$label/${stage}_run/${stage}_stability_p8_locked_panel
    key=${label}_${stage}; arm=ronpo_os_${label}
    mkdir -p "$OUT/generations/$key"
    ln -sfn "$src/generations/$arm/output_42.json" "$OUT/generations/$key/output_42.json"
    ln -sfn "$src/gates/$arm.json" "$OUT/gates/$key.json"
  done
done
models=()
for label in k0p01 k0p1 k0p5; do for stage in 1 2 3 4; do models+=("${label}_stage${stage}"); done; done
for label in k1 k2; do for stage in 1 2 3; do models+=("${label}_stage${stage}"); done; done
csv=$(IFS=,; echo "base,${models[*]}")
"$VENV/bin/python" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/merge_eval_pool.py" \
  --generation-root "$OUT/generations" --models "$csv" --seed 42 --expected-records 1000 \
  --gate-root "$OUT/gates" --output "$OUT/response_pool.jsonl" --audit "$OUT/pool_audit.json"
"$VENV/bin/python" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py" split \
  --input "$OUT/response_pool.jsonl" --output-dir "$OUT/shards" --num-shards 2 --expected-records 1000
pids=()
for gpu in 0 1; do
  flock -x "/tmp/ronpo2_figure_gpu${gpu}.lock" \
    bash "$PROJECT/analysis/p11_saferlhf_stage_curve_20260718/score_worker.sh" "$PROJECT" "$OUT" "$gpu" "$gpu" \
    > "$OUT/logs/score_${gpu}.log" 2>&1 & pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
for objective in helpfulness harmlessness; do
  "$VENV/bin/python" "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py" merge \
    --inputs "$OUT/score_shards/${objective}_0.jsonl" "$OUT/score_shards/${objective}_1.jsonl" \
    --output "$OUT/scores/$objective.jsonl" --audit "$OUT/scores/${objective}_audit.json" \
    --expected-records 1000 --expected-scores-per-row 19 --strip-responses
done
date -Is > "$OUT/COMPLETE"
