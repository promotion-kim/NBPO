#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${USE_EXT_CACHE:-1}" == "1" ]]; then
  # shellcheck source=../scripts/setup_ext_cache.sh
  source "$PROJECT_ROOT/scripts/setup_ext_cache.sh"
fi

PYTHON_TRAIN="${PYTHON_TRAIN:-/home/sjkim/anaconda3/envs/mnpo_train/bin/python}"
PYTHON_INFER="${PYTHON_INFER:-/home/sjkim/anaconda3/envs/mnpo_infer/bin/python}"
PYTHON_RM_SKYWORK="${PYTHON_RM_SKYWORK:-$PYTHON_INFER}"
PYTHON_RM_ATHENE="${PYTHON_RM_ATHENE:-$PYTHON_INFER}"
PYTHON_RM_ARMO="${PYTHON_RM_ARMO:-$PYTHON_TRAIN}"

STAGE="${STAGE:-1}"
EVAL_FILE="${EVAL_FILE:-$PROJECT_ROOT/data/gemma2_ufb_part1_test.jsonl}"
WORK_DIR="${WORK_DIR:-/ext_hdd/sjkim/mnpo/eval/htmnpo_stage${STAGE}_base_compare}"
GEN_DIR="${GEN_DIR:-$WORK_DIR/generations}"
SCORED_DIR="${SCORED_DIR:-$WORK_DIR/scored}"
RESULT_DIR="${RESULT_DIR:-$WORK_DIR/results}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/ext_hdd/sjkim/mnpo/outputs}"

BASELINE_MODEL="${BASELINE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
HT_MODEL_NAME="${HT_MODEL_NAME:-htmnpo_skywork}"
HT_MODEL="${HT_MODEL:-$OUTPUT_ROOT/qwen2.5-1.5b-instruct_htmnpo_skywork_online_multiobj_stage_${STAGE}}"

OBJECTIVES="${OBJECTIVES:-skywork,athene,armo}"
SEED="${SEED:-42}"
DECODE_GPUS="${DECODE_GPUS:-1}"
DECODE_ATTENTION_BACKEND="${DECODE_ATTENTION_BACKEND:-XFORMERS}"
DECODE_TEMPERATURE="${DECODE_TEMPERATURE:-0.7}"
DECODE_TOP_P="${DECODE_TOP_P:-0.9}"
DECODE_MAX_TOKENS="${DECODE_MAX_TOKENS:-2048}"
RM_BATCH_SIZE="${RM_BATCH_SIZE:-16}"
RM_SAMPLE_BATCH_SIZE="${RM_SAMPLE_BATCH_SIZE:-32}"
RM_MAX_SEQ_LENGTH="${RM_MAX_SEQ_LENGTH:-4096}"
RM_MAX_SAMPLES="${RM_MAX_SAMPLES:-}"

export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

if [[ ! -f "$EVAL_FILE" ]]; then
  echo "Missing eval file: $EVAL_FILE" >&2
  exit 1
fi
if [[ ! -d "$HT_MODEL" ]]; then
  echo "Missing HT-MNPO model directory: $HT_MODEL" >&2
  exit 1
fi

mkdir -p "$GEN_DIR" "$SCORED_DIR" "$RESULT_DIR"

decode_model() {
  local name="$1"
  local model="$2"
  local out_dir="$GEN_DIR/$name"
  if [[ "${FORCE_DECODE:-0}" == "1" || ! -f "$out_dir/output_${SEED}.json" ]]; then
    local cache_args=()
    if [[ -n "${CACHE_DIR:-}" ]]; then
      cache_args=(--cache_dir "$CACHE_DIR")
    fi
    local backend_args=()
    if [[ -n "$DECODE_ATTENTION_BACKEND" ]]; then
      backend_args=(--attention_backend "$DECODE_ATTENTION_BACKEND")
    fi
    local sanity_args=()
    if [[ "${EVAL_SANITY_CHECK:-0}" == "1" ]]; then
      sanity_args=(--sanity_check)
    fi
    "$PYTHON_INFER" -u -m on_policy_data_gen.decode \
      --data_dir "$EVAL_FILE" \
      --model "$model" \
      --seeds "$SEED" \
      --output_dir "$out_dir" \
      --num_gpu "$DECODE_GPUS" \
      --temperature "$DECODE_TEMPERATURE" \
      --top_p "$DECODE_TOP_P" \
      --max_tokens "$DECODE_MAX_TOKENS" \
      "${backend_args[@]}" \
      "${cache_args[@]}" \
      "${sanity_args[@]}"
  else
    echo "Skipping decode for $name; found $out_dir/output_${SEED}.json"
  fi
}

score_objectives() {
  local input_json="$1"
  IFS=',' read -ra names <<< "$OBJECTIVES"
  for obj in "${names[@]}"; do
    local output_file="$SCORED_DIR/eval_${obj}.jsonl"
    if [[ "${FORCE_SCORE:-0}" != "1" && -f "$output_file" ]]; then
      echo "Skipping scoring for $obj; found $output_file"
      continue
    fi
    local max_samples_args=()
    if [[ -n "$RM_MAX_SAMPLES" ]]; then
      max_samples_args=(--max_samples "$RM_MAX_SAMPLES")
    fi
    case "$obj" in
      skywork)
        "$PYTHON_RM_SKYWORK" -u -m on_policy_data_gen.rm_skywork \
          --input_file "$input_json" \
          --output_file "$output_file" \
          --cache_dir "${CACHE_DIR:-/cache}" \
          --batch_size "$RM_BATCH_SIZE" \
          --sample_batch_size "$RM_SAMPLE_BATCH_SIZE" \
          --max_seq_length "$RM_MAX_SEQ_LENGTH" \
          "${max_samples_args[@]}"
        ;;
      athene)
        "$PYTHON_RM_ATHENE" -u -m on_policy_data_gen.rm_athene \
          --input_file "$input_json" \
          --output_file "$output_file" \
          --cache_dir "${CACHE_DIR:-/cache}" \
          --batch_size "$RM_BATCH_SIZE" \
          --sample_batch_size "$RM_SAMPLE_BATCH_SIZE" \
          "${max_samples_args[@]}"
        ;;
      armo)
        "$PYTHON_RM_ARMO" -u -m on_policy_data_gen.rm_armo \
          --input_file "$input_json" \
          --output_file "$output_file" \
          --cache_dir "${CACHE_DIR:-/cache}" \
          --batch_size "$RM_BATCH_SIZE" \
          --sample_batch_size "$RM_SAMPLE_BATCH_SIZE" \
          --max_seq_length "$RM_MAX_SEQ_LENGTH" \
          "${max_samples_args[@]}"
        ;;
      *)
        echo "Unknown objective '$obj'. Supported: skywork,athene,armo" >&2
        exit 1
        ;;
    esac
  done
}

objective_args() {
  local args=()
  IFS=',' read -ra names <<< "$OBJECTIVES"
  for obj in "${names[@]}"; do
    args+=("${obj}=${SCORED_DIR}/eval_${obj}.jsonl")
  done
  printf '%q ' "${args[@]}"
}

echo "=== HT-MNPO stage ${STAGE}: Base vs ${HT_MODEL_NAME} ==="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Work dir: $WORK_DIR"
echo "Eval file: $EVAL_FILE"

decode_model baseline "$BASELINE_MODEL"
decode_model "$HT_MODEL_NAME" "$HT_MODEL"

"$PYTHON_TRAIN" -m mnpo_scripts.merge_model_generations \
  --generations \
    "baseline=$GEN_DIR/baseline/output_${SEED}.json" \
    "$HT_MODEL_NAME=$GEN_DIR/$HT_MODEL_NAME/output_${SEED}.json" \
  --output_file "$WORK_DIR/merged_model_generations.json"

score_objectives "$WORK_DIR/merged_model_generations.json"

# shellcheck disable=SC2046
"$PYTHON_TRAIN" -m mnpo_scripts.evaluate_multi_objective_models \
  --scored_files $(objective_args) \
  --output_dir "$RESULT_DIR" \
  --baseline_model baseline

"$PYTHON_TRAIN" -m mnpo_scripts.write_mnpo_benchmark_table \
  --result_dir "$RESULT_DIR" \
  --output_prefix "$RESULT_DIR/base_vs_${HT_MODEL_NAME}_stage${STAGE}" \
  --objectives "$OBJECTIVES"

echo "Evaluation written to $RESULT_DIR"
