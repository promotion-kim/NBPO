#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=setup_ext_cache.sh
source "$PROJECT_ROOT/scripts/setup_ext_cache.sh"

PYTHON_TRAIN="${PYTHON_TRAIN:-/home/sjkim/anaconda3/envs/mnpo_train/bin/python}"
PYTHON_INFER="${PYTHON_INFER:-/home/sjkim/anaconda3/envs/mnpo_infer/bin/python}"
PYTHON_RM_SKYWORK="${PYTHON_RM_SKYWORK:-$PYTHON_INFER}"
PYTHON_RM_ATHENE="${PYTHON_RM_ATHENE:-$PYTHON_INFER}"
PYTHON_RM_ARMO="${PYTHON_RM_ARMO:-$PYTHON_TRAIN}"

EVAL_FILE="${EVAL_FILE:-$PROJECT_ROOT/data/gemma2_ufb_part2_test.jsonl}"
WORK_DIR="${WORK_DIR:-/ext_hdd/sjkim/mnpo/eval/htmnpo_ronpo_stage2_base_compare}"
GEN_DIR="$WORK_DIR/generations"
SCORED_DIR="$WORK_DIR/scored"
RESULT_DIR="$WORK_DIR/results"
LOG_DIR="$WORK_DIR/logs"
SEED="${SEED:-42}"
DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE:-512}"
DECODE_ATTENTION_BACKEND="${DECODE_ATTENTION_BACKEND:-XFORMERS}"
DECODE_DTYPE="${DECODE_DTYPE:-bfloat16}"
RM_SKYWORK_BATCH_SIZE="${RM_SKYWORK_BATCH_SIZE:-4}"
RM_SKYWORK_SAMPLE_BATCH_SIZE="${RM_SKYWORK_SAMPLE_BATCH_SIZE:-8}"
RM_SKYWORK_ATTN_IMPLEMENTATION="${RM_SKYWORK_ATTN_IMPLEMENTATION:-sdpa}"
RM_ATHENE_BATCH_SIZE="${RM_ATHENE_BATCH_SIZE:-4}"
RM_ATHENE_SAMPLE_BATCH_SIZE="${RM_ATHENE_SAMPLE_BATCH_SIZE:-8}"
RM_ARMO_BATCH_SIZE="${RM_ARMO_BATCH_SIZE:-8}"
RM_ARMO_SAMPLE_BATCH_SIZE="${RM_ARMO_SAMPLE_BATCH_SIZE:-8}"

BASELINE_MODEL="${BASELINE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
HT_SKYWORK_MODEL="${HT_SKYWORK_MODEL:-/ext_hdd/sjkim/mnpo/ht_stage1_out/qwen2.5-1.5b-instruct_htmnpo_skywork_online_multiobj_stage_2}"
HT_ATHENE_MODEL="${HT_ATHENE_MODEL:-/ext_hdd/sjkim/mnpo/ht_stage1_out/qwen2.5-1.5b-instruct_htmnpo_athene_online_multiobj_stage_2}"
HT_ARMO_MODEL="${HT_ARMO_MODEL:-/ext_hdd/sjkim/mnpo/ht_stage1_out/qwen2.5-1.5b-instruct_htmnpo_armo_online_multiobj_stage_2}"
RONPO_MODEL="${RONPO_MODEL:-/ext_hdd/sjkim/mnpo/outputs_ronpo_h200/qwen2.5-1.5b-instruct_ronpo_vs_htmnpo_online_multiobj_stage_2}"

export PYTHONPATH="$PROJECT_ROOT"
mkdir -p "$GEN_DIR" "$SCORED_DIR" "$RESULT_DIR" "$LOG_DIR"

require_path() {
  local path="$1"
  if [[ "$path" == Qwen/* ]]; then
    return
  fi
  test -d "$path"
}

run_decode() {
  local name="$1"
  local model="$2"
  local gpu="$3"
  local out_dir="$GEN_DIR/$name"
  local out_file="$out_dir/output_${SEED}.json"
  mkdir -p "$out_dir"
  if [[ "${FORCE_DECODE:-0}" != "1" && -f "$out_file" ]]; then
    echo "[decode:$name] skip existing $out_file"
    return
  fi
  echo "[decode:$name] gpu=$gpu model=$model"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_INFER" -u -m on_policy_data_gen.decode \
    --data_dir "$EVAL_FILE" \
    --model "$model" \
    --seeds "$SEED" \
    --output_dir "$out_dir" \
    --num_gpu 1 \
    --temperature 0.7 \
    --top_p 0.9 \
    --batch_size "$DECODE_BATCH_SIZE" \
    --attention_backend "$DECODE_ATTENTION_BACKEND" \
    --dtype "$DECODE_DTYPE" \
    --cache_dir "$CACHE_DIR" \
    >"$LOG_DIR/decode_${name}.log" 2>&1
}

wait_batch() {
  local pids=("$@")
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    echo "At least one background job failed. See $LOG_DIR." >&2
    exit 1
  fi
}

expected_samples() {
  "$PYTHON_TRAIN" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' \
    "$WORK_DIR/merged_model_generations.json"
}

score_complete() {
  local output_file="$1"
  [[ -s "$output_file" ]] || return 1
  local got expected
  got="$(wc -l < "$output_file")"
  expected="$(expected_samples)"
  [[ "$got" == "$expected" ]]
}

score_skywork() {
  local output_file="$SCORED_DIR/eval_skywork.jsonl"
  if [[ "${FORCE_SCORE:-0}" != "1" ]] && score_complete "$output_file"; then
    echo "[score:skywork] skip existing $output_file"
    return
  fi
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_RM_SKYWORK" -u -m on_policy_data_gen.rm_skywork \
    --input_file "$WORK_DIR/merged_model_generations.json" \
    --output_file "$output_file" \
    --cache_dir "$CACHE_DIR" \
    --batch_size "$RM_SKYWORK_BATCH_SIZE" \
    --sample_batch_size "$RM_SKYWORK_SAMPLE_BATCH_SIZE" \
    --attn_implementation "$RM_SKYWORK_ATTN_IMPLEMENTATION" \
    >"$LOG_DIR/score_skywork.log" 2>&1
}

score_athene() {
  local output_file="$SCORED_DIR/eval_athene.jsonl"
  if [[ "${FORCE_SCORE:-0}" != "1" ]] && score_complete "$output_file"; then
    echo "[score:athene] skip existing $output_file"
    return
  fi
  CUDA_VISIBLE_DEVICES=1 "$PYTHON_RM_ATHENE" -u -m on_policy_data_gen.rm_athene \
    --input_file "$WORK_DIR/merged_model_generations.json" \
    --output_file "$output_file" \
    --cache_dir "$CACHE_DIR" \
    --batch_size "$RM_ATHENE_BATCH_SIZE" \
    --sample_batch_size "$RM_ATHENE_SAMPLE_BATCH_SIZE" \
    >"$LOG_DIR/score_athene.log" 2>&1
}

score_armo() {
  local output_file="$SCORED_DIR/eval_armo.jsonl"
  if [[ "${FORCE_SCORE:-0}" != "1" ]] && score_complete "$output_file"; then
    echo "[score:armo] skip existing $output_file"
    return
  fi
  CUDA_VISIBLE_DEVICES=2 "$PYTHON_RM_ARMO" -u -m on_policy_data_gen.rm_armo \
    --input_file "$WORK_DIR/merged_model_generations.json" \
    --output_file "$output_file" \
    --cache_dir "$CACHE_DIR" \
    --batch_size "$RM_ARMO_BATCH_SIZE" \
    --sample_batch_size "$RM_ARMO_SAMPLE_BATCH_SIZE" \
    >"$LOG_DIR/score_armo.log" 2>&1
}

echo "[start] stage-2 controlled local RM evaluation at $(date -Is)"
echo "[work] $WORK_DIR"
echo "[eval] $EVAL_FILE"
for model_path in "$HT_SKYWORK_MODEL" "$HT_ATHENE_MODEL" "$HT_ARMO_MODEL" "$RONPO_MODEL"; do
  require_path "$model_path"
done

run_decode baseline "$BASELINE_MODEL" 0 &
p1=$!
run_decode htmnpo_skywork "$HT_SKYWORK_MODEL" 1 &
p2=$!
run_decode htmnpo_athene "$HT_ATHENE_MODEL" 2 &
p3=$!
wait_batch "$p1" "$p2" "$p3"

run_decode htmnpo_armo "$HT_ARMO_MODEL" 0 &
p1=$!
run_decode ronpo "$RONPO_MODEL" 1 &
p2=$!
wait_batch "$p1" "$p2"

"$PYTHON_TRAIN" -m mnpo_scripts.merge_model_generations \
  --generations \
    "baseline=$GEN_DIR/baseline/output_${SEED}.json" \
    "htmnpo_skywork=$GEN_DIR/htmnpo_skywork/output_${SEED}.json" \
    "htmnpo_athene=$GEN_DIR/htmnpo_athene/output_${SEED}.json" \
    "htmnpo_armo=$GEN_DIR/htmnpo_armo/output_${SEED}.json" \
    "ronpo=$GEN_DIR/ronpo/output_${SEED}.json" \
  --output_file "$WORK_DIR/merged_model_generations.json" \
  >"$LOG_DIR/merge.log" 2>&1

score_skywork &
p1=$!
score_athene &
p2=$!
score_armo &
p3=$!
wait_batch "$p1" "$p2" "$p3"

"$PYTHON_TRAIN" -m mnpo_scripts.evaluate_multi_objective_models \
  --scored_files \
    "skywork=$SCORED_DIR/eval_skywork.jsonl" \
    "athene=$SCORED_DIR/eval_athene.jsonl" \
    "armo=$SCORED_DIR/eval_armo.jsonl" \
  --output_dir "$RESULT_DIR" \
  --baseline_model baseline \
  >"$LOG_DIR/evaluate.log" 2>&1

echo "[done] results written to $RESULT_DIR at $(date -Is)"
