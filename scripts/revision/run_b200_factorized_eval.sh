#!/usr/bin/env bash
# Controlled 6-policy eval of the factorized-adversary ablation on a B200 host.
# Waits for the three B200 arms, reuses the frozen 20260716 generations for
# baseline/topmass/os (identical responses; only the scorer host changes, and
# all six policies are scored by the same B200 stack so the set is internally
# consistent), decodes the new arms, scores with the three RMs, aggregates.
# Run on ronpo (4 free GPUs after the arms finish).
set -euo pipefail
SJ=/NHNHOME/AIPR/sjkim
PROJECT_ROOT="$SJ/MNPO_rev_20260720"
WORK="$SJ/b200_eval_20260720"
ARMS="$SJ/ronpo_arms_20260720"
PY="$SJ/venv_clean/bin/python"
EVAL_FILE="$PROJECT_ROOT/data/gemma2_ufb_part2_test.jsonl"
SEED=42

export PYTHONPATH="$PROJECT_ROOT"
export HF_HOME="$SJ/baseline_repair_1p5b_20260714/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$WORK"/{generations,scored,results,logs}

declare -A NEW=(
  [os_b200]="$ARMS/os-ronpo-os-k05-b200"
  [konly_b200]="$ARMS/os-ronpo-konly-k05-b200"
  [aonly_b200]="$ARMS/os-ronpo-aonly-k05-b200"
)

echo "[eval] waiting for arm checkpoints"
for name in "${!NEW[@]}"; do
  while [[ ! -f "${NEW[$name]}/all_results.json" ]]; do sleep 300; done
  echo "[eval] ${name} ready"
done

for name in baseline topmass os; do
  mkdir -p "$WORK/generations/$name"
  cp -n "$WORK/frozen_gens/$name/output_${SEED}.json" "$WORK/generations/$name/" 2>/dev/null || true
done

gpu=0
pids=()
for name in "${!NEW[@]}"; do
  out="$WORK/generations/$name"
  mkdir -p "$out"
  if [[ ! -f "$out/output_${SEED}.json" ]]; then
    echo "[decode:$name] gpu=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu "$PY" -u -m on_policy_data_gen.decode \
      --data_dir "$EVAL_FILE" --model "${NEW[$name]}" --seeds $SEED --output_dir "$out" \
      --num_gpu 1 --temperature 0.7 --top_p 0.9 --batch_size 512 \
      --max_tokens 4096 --dtype bfloat16 --cache_dir "$HF_HUB_CACHE" \
      > "$WORK/logs/decode_$name.log" 2>&1 &
    pids+=($!)
  fi
  gpu=$((gpu + 1))
done
for p in "${pids[@]:-}"; do [[ -n "$p" ]] && wait "$p"; done

GEN_ARGS=()
for d in "$WORK"/generations/*/; do
  name="$(basename "$d")"
  GEN_ARGS+=("$name=${d}output_${SEED}.json")
done
"$PY" -m mnpo_scripts.merge_model_generations \
  --generations "${GEN_ARGS[@]}" \
  --output_file "$WORK/merged_model_generations.json" > "$WORK/logs/merge.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "$PY" -u -m on_policy_data_gen.rm_skywork \
  --input_file "$WORK/merged_model_generations.json" --output_file "$WORK/scored/eval_skywork.jsonl" \
  --cache_dir "$HF_HUB_CACHE" --batch_size 8 --sample_batch_size 32 \
  > "$WORK/logs/score_skywork.log" 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=1 "$PY" -u -m on_policy_data_gen.rm_athene \
  --input_file "$WORK/merged_model_generations.json" --output_file "$WORK/scored/eval_athene.jsonl" \
  --cache_dir "$HF_HUB_CACHE" --batch_size 8 --sample_batch_size 32 \
  > "$WORK/logs/score_athene.log" 2>&1 &
p2=$!
CUDA_VISIBLE_DEVICES=2 "$PY" -u -m on_policy_data_gen.rm_armo \
  --input_file "$WORK/merged_model_generations.json" --output_file "$WORK/scored/eval_armo.jsonl" \
  --cache_dir "$HF_HUB_CACHE" --batch_size 8 --sample_batch_size 32 \
  > "$WORK/logs/score_armo.log" 2>&1 &
p3=$!
wait $p1 $p2 $p3

"$PY" -m mnpo_scripts.evaluate_multi_objective_models \
  --scored_files "skywork=$WORK/scored/eval_skywork.jsonl" "athene=$WORK/scored/eval_athene.jsonl" "armo=$WORK/scored/eval_armo.jsonl" \
  --output_dir "$WORK/results" --baseline_model baseline > "$WORK/logs/evaluate.log" 2>&1
echo "[eval] done: $WORK/results"
cat "$WORK/results/model_summary.json" 2>/dev/null | head -60
