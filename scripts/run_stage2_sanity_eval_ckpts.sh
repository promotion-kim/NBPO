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
WORK_DIR="${WORK_DIR:-/ext_hdd/sjkim/mnpo/eval/ronpo_stage2_resume_sanity_20260625}"
GEN_DIR="$WORK_DIR/generations"
SCORED_DIR="$WORK_DIR/scored"
RESULT_DIR="$WORK_DIR/results"
LOG_DIR="$WORK_DIR/logs"
SEED="${SEED:-42}"

DECODE_GPUS_CSV="${DECODE_GPUS_CSV:-1,2}"
SCORE_GPU_SKYWORK="${SCORE_GPU_SKYWORK:-1}"
SCORE_GPU_ATHENE="${SCORE_GPU_ATHENE:-2}"
SCORE_GPU_ARMO="${SCORE_GPU_ARMO:-1}"
DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE:-512}"
DECODE_ATTENTION_BACKEND="${DECODE_ATTENTION_BACKEND:-XFORMERS}"
DECODE_DTYPE="${DECODE_DTYPE:-bfloat16}"
MAX_TOKENS="${MAX_TOKENS:-4096}"

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
RONPO_S2_CKPT1400_MODEL="${RONPO_S2_CKPT1400_MODEL:-/ext_hdd/sjkim/mnpo/outputs_ronpo_h200/qwen2.5-1.5b-instruct_ronpo_stage2_relative_lr2e8_od2g2/checkpoint-1400}"
RONPO_S2_CKPT2457_MODEL="${RONPO_S2_CKPT2457_MODEL:-/ext_hdd/sjkim/mnpo/outputs_ronpo_h200/qwen2.5-1.5b-instruct_ronpo_stage2_relative_lr2e8_od2g2/checkpoint-2457}"

export PYTHONPATH="$PROJECT_ROOT"
mkdir -p "$GEN_DIR" "$SCORED_DIR" "$RESULT_DIR" "$LOG_DIR"

IFS=',' read -r -a DECODE_GPUS <<< "$DECODE_GPUS_CSV"
if [[ "${#DECODE_GPUS[@]}" -lt 1 ]]; then
  echo "DECODE_GPUS_CSV must contain at least one GPU id." >&2
  exit 1
fi

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
    --max_tokens "$MAX_TOKENS" \
    --batch_size "$DECODE_BATCH_SIZE" \
    --attention_backend "$DECODE_ATTENTION_BACKEND" \
    --dtype "$DECODE_DTYPE" \
    --cache_dir "$CACHE_DIR" \
    >"$LOG_DIR/decode_${name}.log" 2>&1
}

wait_batch() {
  local failed=0
  for pid in "$@"; do
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
  CUDA_VISIBLE_DEVICES="$SCORE_GPU_SKYWORK" "$PYTHON_RM_SKYWORK" -u -m on_policy_data_gen.rm_skywork \
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
  CUDA_VISIBLE_DEVICES="$SCORE_GPU_ATHENE" "$PYTHON_RM_ATHENE" -u -m on_policy_data_gen.rm_athene \
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
  CUDA_VISIBLE_DEVICES="$SCORE_GPU_ARMO" "$PYTHON_RM_ARMO" -u -m on_policy_data_gen.rm_armo \
    --input_file "$WORK_DIR/merged_model_generations.json" \
    --output_file "$output_file" \
    --cache_dir "$CACHE_DIR" \
    --batch_size "$RM_ARMO_BATCH_SIZE" \
    --sample_batch_size "$RM_ARMO_SAMPLE_BATCH_SIZE" \
    >"$LOG_DIR/score_armo.log" 2>&1
}

echo "[start] RONPO stage-2 resume sanity evaluation at $(date -Is)"
echo "[work] $WORK_DIR"
echo "[eval] $EVAL_FILE"
echo "[decode_gpus] $DECODE_GPUS_CSV"

for model_path in \
  "$HT_SKYWORK_MODEL" \
  "$HT_ATHENE_MODEL" \
  "$HT_ARMO_MODEL" \
  "$RONPO_S2_CKPT1400_MODEL" \
  "$RONPO_S2_CKPT2457_MODEL"; do
  require_path "$model_path"
done

MODEL_NAMES=(
  baseline
  htmnpo_skywork_s2
  htmnpo_athene_s2
  htmnpo_armo_s2
  ronpo_s2_ckpt1400
  ronpo_s2_ckpt2457
)
MODEL_PATHS=(
  "$BASELINE_MODEL"
  "$HT_SKYWORK_MODEL"
  "$HT_ATHENE_MODEL"
  "$HT_ARMO_MODEL"
  "$RONPO_S2_CKPT1400_MODEL"
  "$RONPO_S2_CKPT2457_MODEL"
)

idx=0
while [[ "$idx" -lt "${#MODEL_NAMES[@]}" ]]; do
  pids=()
  for offset in "${!DECODE_GPUS[@]}"; do
    model_idx=$((idx + offset))
    [[ "$model_idx" -lt "${#MODEL_NAMES[@]}" ]] || continue
    run_decode "${MODEL_NAMES[$model_idx]}" "${MODEL_PATHS[$model_idx]}" "${DECODE_GPUS[$offset]}" &
    pids+=("$!")
  done
  wait_batch "${pids[@]}"
  idx=$((idx + ${#DECODE_GPUS[@]}))
done

"$PYTHON_TRAIN" -m mnpo_scripts.merge_model_generations \
  --generations \
    "baseline=$GEN_DIR/baseline/output_${SEED}.json" \
    "htmnpo_skywork_s2=$GEN_DIR/htmnpo_skywork_s2/output_${SEED}.json" \
    "htmnpo_athene_s2=$GEN_DIR/htmnpo_athene_s2/output_${SEED}.json" \
    "htmnpo_armo_s2=$GEN_DIR/htmnpo_armo_s2/output_${SEED}.json" \
    "ronpo_s2_ckpt1400=$GEN_DIR/ronpo_s2_ckpt1400/output_${SEED}.json" \
    "ronpo_s2_ckpt2457=$GEN_DIR/ronpo_s2_ckpt2457/output_${SEED}.json" \
  --output_file "$WORK_DIR/merged_model_generations.json" \
  >"$LOG_DIR/merge.log" 2>&1

"$PYTHON_TRAIN" -m mnpo_scripts.analyze_generation_quality \
  --merged_file "$WORK_DIR/merged_model_generations.json" \
  --output_dir "$RESULT_DIR" \
  >"$LOG_DIR/generation_quality.log" 2>&1

score_skywork &
p1=$!
score_athene &
p2=$!
wait_batch "$p1" "$p2"

score_armo

"$PYTHON_TRAIN" -m mnpo_scripts.evaluate_multi_objective_models \
  --scored_files \
    "skywork=$SCORED_DIR/eval_skywork.jsonl" \
    "athene=$SCORED_DIR/eval_athene.jsonl" \
    "armo=$SCORED_DIR/eval_armo.jsonl" \
  --output_dir "$RESULT_DIR" \
  --baseline_model baseline \
  >"$LOG_DIR/evaluate.log" 2>&1

echo "[done] results written to $RESULT_DIR at $(date -Is)"
