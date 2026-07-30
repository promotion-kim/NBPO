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
SOURCE_WORK_DIR="${SOURCE_WORK_DIR:-/ext_hdd/sjkim/mnpo/eval/htmnpo_ronpo_stage2_base_compare}"
WORK_DIR="${WORK_DIR:-/ext_hdd/sjkim/mnpo/eval/ronpo_stage2_ckpt_compare_20260624}"
GEN_DIR="$WORK_DIR/generations"
SCORED_DIR="$WORK_DIR/scored"
RESULT_DIR="$WORK_DIR/results"
LOG_DIR="$WORK_DIR/logs"

SEED="${SEED:-42}"
GPU="${GPU:-0}"
DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE:-64}"
DECODE_ATTENTION_BACKEND="${DECODE_ATTENTION_BACKEND:-XFORMERS}"
DECODE_DTYPE="${DECODE_DTYPE:-bfloat16}"
DECODE_GPU_MEMORY_UTILIZATION="${DECODE_GPU_MEMORY_UTILIZATION:-0.55}"

RM_SKYWORK_BATCH_SIZE="${RM_SKYWORK_BATCH_SIZE:-4}"
RM_SKYWORK_SAMPLE_BATCH_SIZE="${RM_SKYWORK_SAMPLE_BATCH_SIZE:-8}"
RM_SKYWORK_ATTN_IMPLEMENTATION="${RM_SKYWORK_ATTN_IMPLEMENTATION:-sdpa}"
RM_ATHENE_BATCH_SIZE="${RM_ATHENE_BATCH_SIZE:-4}"
RM_ATHENE_SAMPLE_BATCH_SIZE="${RM_ATHENE_SAMPLE_BATCH_SIZE:-8}"
RM_ARMO_BATCH_SIZE="${RM_ARMO_BATCH_SIZE:-8}"
RM_ARMO_SAMPLE_BATCH_SIZE="${RM_ARMO_SAMPLE_BATCH_SIZE:-8}"

RONPO_LR5E8_CKPT="${RONPO_LR5E8_CKPT:-/ext_hdd/sjkim/mnpo/outputs_ronpo_h200/qwen2.5-1.5b-instruct_ronpo_stage2_relative_lr5e8_odin2/checkpoint-500}"
RONPO_LR2E8_CKPT="${RONPO_LR2E8_CKPT:-/ext_hdd/sjkim/mnpo/outputs_ronpo_h200/qwen2.5-1.5b-instruct_ronpo_stage2_relative_lr2e8_od2g2/checkpoint-300}"

export PYTHONPATH="$PROJECT_ROOT"
mkdir -p "$GEN_DIR" "$SCORED_DIR" "$RESULT_DIR" "$LOG_DIR"

require_file() {
  test -f "$1"
}

require_dir() {
  test -d "$1"
}

decode_model() {
  local name="$1"
  local model="$2"
  local out_dir="$GEN_DIR/$name"
  local out_file="$out_dir/output_${SEED}.json"
  mkdir -p "$out_dir"
  if [[ "${FORCE_DECODE:-0}" != "1" && -s "$out_file" ]]; then
    echo "[decode:$name] skip existing $out_file"
    return
  fi
  echo "[decode:$name] gpu=$GPU model=$model"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_INFER" -u -m on_policy_data_gen.decode \
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
    --gpu_memory_utilization "$DECODE_GPU_MEMORY_UTILIZATION" \
    --cache_dir "$CACHE_DIR" \
    >"$LOG_DIR/decode_${name}.log" 2>&1
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
  echo "[score:skywork] gpu=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_RM_SKYWORK" -u -m on_policy_data_gen.rm_skywork \
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
  echo "[score:athene] gpu=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_RM_ATHENE" -u -m on_policy_data_gen.rm_athene \
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
  echo "[score:armo] gpu=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_RM_ARMO" -u -m on_policy_data_gen.rm_armo \
    --input_file "$WORK_DIR/merged_model_generations.json" \
    --output_file "$output_file" \
    --cache_dir "$CACHE_DIR" \
    --batch_size "$RM_ARMO_BATCH_SIZE" \
    --sample_batch_size "$RM_ARMO_SAMPLE_BATCH_SIZE" \
    >"$LOG_DIR/score_armo.log" 2>&1
}

echo "[start] RONPO stage-2 checkpoint reward eval $(date -Is)"
echo "[work] $WORK_DIR"
echo "[gpu] $GPU"
echo "[source] $SOURCE_WORK_DIR"

for f in \
  "$SOURCE_WORK_DIR/generations/baseline/output_${SEED}.json" \
  "$SOURCE_WORK_DIR/generations/htmnpo_skywork/output_${SEED}.json" \
  "$SOURCE_WORK_DIR/generations/htmnpo_athene/output_${SEED}.json" \
  "$SOURCE_WORK_DIR/generations/htmnpo_armo/output_${SEED}.json" \
  "$SOURCE_WORK_DIR/generations/ronpo/output_${SEED}.json"; do
  require_file "$f"
done
require_dir "$RONPO_LR5E8_CKPT"
require_dir "$RONPO_LR2E8_CKPT"

decode_model ronpo_lr5e8_s500 "$RONPO_LR5E8_CKPT"
decode_model ronpo_lr2e8_s300 "$RONPO_LR2E8_CKPT"

"$PYTHON_TRAIN" -m mnpo_scripts.merge_model_generations \
  --generations \
    "baseline=$SOURCE_WORK_DIR/generations/baseline/output_${SEED}.json" \
    "htmnpo_skywork_s2=$SOURCE_WORK_DIR/generations/htmnpo_skywork/output_${SEED}.json" \
    "htmnpo_athene_s2=$SOURCE_WORK_DIR/generations/htmnpo_athene/output_${SEED}.json" \
    "htmnpo_armo_s2=$SOURCE_WORK_DIR/generations/htmnpo_armo/output_${SEED}.json" \
    "ronpo_s2_old=$SOURCE_WORK_DIR/generations/ronpo/output_${SEED}.json" \
    "ronpo_lr5e8_s500=$GEN_DIR/ronpo_lr5e8_s500/output_${SEED}.json" \
    "ronpo_lr2e8_s300=$GEN_DIR/ronpo_lr2e8_s300/output_${SEED}.json" \
  --output_file "$WORK_DIR/merged_model_generations.json" \
  >"$LOG_DIR/merge.log" 2>&1

score_skywork
score_athene
score_armo

"$PYTHON_TRAIN" -m mnpo_scripts.evaluate_multi_objective_models \
  --scored_files \
    "skywork=$SCORED_DIR/eval_skywork.jsonl" \
    "athene=$SCORED_DIR/eval_athene.jsonl" \
    "armo=$SCORED_DIR/eval_armo.jsonl" \
  --output_dir "$RESULT_DIR" \
  --baseline_model baseline \
  >"$LOG_DIR/evaluate.log" 2>&1

echo "[done] results written to $RESULT_DIR $(date -Is)"
