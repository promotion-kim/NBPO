#!/usr/bin/env bash
# Controlled local-RM eval of new stage-2 arms against the shipped set.
# Reuses the frozen 20260716 generations for baseline/topmass/os and decodes
# only the new arms, then scores and aggregates everything jointly (the
# aggregator renormalizes per prompt within this exact model set).
#   NEW_MODELS="konly=/path aonly=/path" WORK_DIR=... GPUS="1 2" bash eval_factorized_arms.sh
set -euo pipefail
cd /home/sjkim/MNPO
source /home/sjkim/MNPO/scripts/setup_ext_cache.sh

PYTHON_TRAIN=/home/sjkim/anaconda3/envs/mnpo_train/bin/python
PYTHON_INFER=/home/sjkim/anaconda3/envs/mnpo_infer/bin/python
EVAL_FILE=/home/sjkim/MNPO/data/gemma2_ufb_part2_test.jsonl
OLD_GEN=/ext_hdd/sjkim/mnpo/eval/ronpo_topmass_vs_os_stage2_20260716/generations
WORK_DIR="${WORK_DIR:?}"
NEW_MODELS="${NEW_MODELS:?space-separated name=path}"
REUSE="${REUSE:-baseline topmass os}"
read -ra GPU_ARR <<< "${GPUS:-1 2}"
SEED=42
export PYTHONPATH=/home/sjkim/MNPO

mkdir -p "$WORK_DIR"/{generations,scored,results,logs}
for name in $REUSE; do
  mkdir -p "$WORK_DIR/generations/$name"
  cp -n "$OLD_GEN/$name/output_${SEED}.json" "$WORK_DIR/generations/$name/" 2>/dev/null || true
done

i=0
pids=()
for spec in $NEW_MODELS; do
  name="${spec%%=*}"; model="${spec#*=}"
  gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
  out="$WORK_DIR/generations/$name"
  mkdir -p "$out"
  if [[ ! -f "$out/output_${SEED}.json" ]]; then
    echo "[decode:$name] gpu=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON_INFER -u -m on_policy_data_gen.decode \
      --data_dir "$EVAL_FILE" --model "$model" --seeds $SEED --output_dir "$out" \
      --num_gpu 1 --temperature 0.7 --top_p 0.9 --batch_size 512 \
      --attention_backend XFORMERS --dtype bfloat16 --cache_dir "$CACHE_DIR" \
      > "$WORK_DIR/logs/decode_$name.log" 2>&1 &
    pids+=($!)
  fi
  i=$((i + 1))
done
for p in "${pids[@]:-}"; do [[ -n "$p" ]] && wait "$p"; done

GEN_ARGS=()
for d in "$WORK_DIR"/generations/*/; do
  name="$(basename "$d")"
  GEN_ARGS+=("$name=$d/output_${SEED}.json")
done
$PYTHON_TRAIN -m mnpo_scripts.merge_model_generations \
  --generations "${GEN_ARGS[@]}" \
  --output_file "$WORK_DIR/merged_model_generations.json" > "$WORK_DIR/logs/merge.log" 2>&1

g0="${GPU_ARR[0]}"; g1="${GPU_ARR[$(( ${#GPU_ARR[@]} > 1 ? 1 : 0 ))]}"
CUDA_VISIBLE_DEVICES=$g0 $PYTHON_INFER -u -m on_policy_data_gen.rm_skywork \
  --input_file "$WORK_DIR/merged_model_generations.json" --output_file "$WORK_DIR/scored/eval_skywork.jsonl" \
  --cache_dir "$CACHE_DIR" --batch_size 4 --sample_batch_size 8 --attn_implementation sdpa \
  > "$WORK_DIR/logs/score_skywork.log" 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=$g1 $PYTHON_INFER -u -m on_policy_data_gen.rm_athene \
  --input_file "$WORK_DIR/merged_model_generations.json" --output_file "$WORK_DIR/scored/eval_athene.jsonl" \
  --cache_dir "$CACHE_DIR" --batch_size 4 --sample_batch_size 8 \
  > "$WORK_DIR/logs/score_athene.log" 2>&1 &
p2=$!
wait $p1 $p2
CUDA_VISIBLE_DEVICES=$g0 $PYTHON_TRAIN -u -m on_policy_data_gen.rm_armo \
  --input_file "$WORK_DIR/merged_model_generations.json" --output_file "$WORK_DIR/scored/eval_armo.jsonl" \
  --cache_dir "$CACHE_DIR" --batch_size 8 --sample_batch_size 8 \
  > "$WORK_DIR/logs/score_armo.log" 2>&1

$PYTHON_TRAIN -m mnpo_scripts.evaluate_multi_objective_models \
  --scored_files "skywork=$WORK_DIR/scored/eval_skywork.jsonl" "athene=$WORK_DIR/scored/eval_athene.jsonl" "armo=$WORK_DIR/scored/eval_armo.jsonl" \
  --output_dir "$WORK_DIR/results" --baseline_model baseline > "$WORK_DIR/logs/evaluate.log" 2>&1
echo "[done] $WORK_DIR/results"
