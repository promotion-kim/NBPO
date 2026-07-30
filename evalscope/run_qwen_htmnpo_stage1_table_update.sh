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
SCORED_DIR="${SCORED_DIR:-$WORK_DIR/scored_extended}"
RESULT_DIR="${RESULT_DIR:-$WORK_DIR/results_extended}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/ext_hdd/sjkim/mnpo/outputs}"

BASELINE_MODEL="${BASELINE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
HT_SKYWORK_MODEL="${HT_SKYWORK_MODEL:-$OUTPUT_ROOT/qwen2.5-1.5b-instruct_htmnpo_skywork_online_multiobj_stage_${STAGE}}"
HT_ATHENE_MODEL="${HT_ATHENE_MODEL:-$OUTPUT_ROOT/qwen2.5-1.5b-instruct_htmnpo_athene_online_multiobj_stage_${STAGE}/checkpoint-300}"
HT_ARMORM_MODEL="${HT_ARMORM_MODEL:-$OUTPUT_ROOT/qwen2.5-1.5b-instruct_htmnpo_armo_online_multiobj_stage_${STAGE}}"
RONPO_OUTPUT_ROOT="${RONPO_OUTPUT_ROOT:-/ext_hdd/sjkim/mnpo/outputs_ronpo_fair}"
RONPO_MODEL="${RONPO_MODEL:-$RONPO_OUTPUT_ROOT/qwen2.5-1.5b-instruct_ronpo_vs_htmnpo_online_multiobj_stage_${STAGE}/checkpoint-1100}"

OBJECTIVES="${OBJECTIVES:-skywork,athene,armo}"
SEED="${SEED:-42}"
DECODE_GPUS="${DECODE_GPUS:-1}"
DECODE_ATTENTION_BACKEND="${DECODE_ATTENTION_BACKEND:-XFORMERS}"
DECODE_TEMPERATURE="${DECODE_TEMPERATURE:-0.7}"
DECODE_TOP_P="${DECODE_TOP_P:-0.9}"
DECODE_MAX_TOKENS="${DECODE_MAX_TOKENS:-2048}"
RM_BATCH_SIZE="${RM_BATCH_SIZE:-8}"
RM_SAMPLE_BATCH_SIZE="${RM_SAMPLE_BATCH_SIZE:-32}"
RM_MAX_SEQ_LENGTH="${RM_MAX_SEQ_LENGTH:-4096}"
INCLUDE_ARMORM="${INCLUDE_ARMORM:-1}"
INCLUDE_RONPO="${INCLUDE_RONPO:-1}"
INCLUDE_SPPO="${INCLUDE_SPPO:-0}"
INCLUDE_INPO="${INCLUDE_INPO:-0}"
SPPO_MODEL="${SPPO_MODEL:-/home/sjkim/mnpo_runs/loki3/out/sppo_s1}"
INPO_MODEL="${INPO_MODEL:-/home/sjkim/mnpo_runs/loki3/out/inpo_s1}"

export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

mkdir -p "$GEN_DIR" "$SCORED_DIR" "$RESULT_DIR"

decode_model() {
  local name="$1"
  local model="$2"
  local out_dir="$GEN_DIR/$name"
  if [[ "${FORCE_DECODE:-0}" == "1" || ! -f "$out_dir/output_${SEED}.json" ]]; then
    if [[ ! -d "$model" && "$model" != Qwen/* ]]; then
      echo "Missing model directory for $name: $model" >&2
      return 1
    fi
    local cache_args=()
    if [[ -n "${CACHE_DIR:-}" ]]; then
      cache_args=(--cache_dir "$CACHE_DIR")
    fi
    local backend_args=()
    if [[ -n "$DECODE_ATTENTION_BACKEND" ]]; then
      backend_args=(--attention_backend "$DECODE_ATTENTION_BACKEND")
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
      "${cache_args[@]}"
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
    case "$obj" in
      skywork)
        "$PYTHON_RM_SKYWORK" -u -m on_policy_data_gen.rm_skywork \
          --input_file "$input_json" \
          --output_file "$output_file" \
          --cache_dir "${CACHE_DIR:-/cache}" \
          --batch_size "$RM_BATCH_SIZE" \
          --sample_batch_size "$RM_SAMPLE_BATCH_SIZE" \
          --max_seq_length "$RM_MAX_SEQ_LENGTH"
        ;;
      athene)
        "$PYTHON_RM_ATHENE" -u -m on_policy_data_gen.rm_athene \
          --input_file "$input_json" \
          --output_file "$output_file" \
          --cache_dir "${CACHE_DIR:-/cache}" \
          --batch_size "$RM_BATCH_SIZE" \
          --sample_batch_size "$RM_SAMPLE_BATCH_SIZE"
        ;;
      armo)
        "$PYTHON_RM_ARMO" -u -m on_policy_data_gen.rm_armo \
          --input_file "$input_json" \
          --output_file "$output_file" \
          --cache_dir "${CACHE_DIR:-/cache}" \
          --batch_size "$RM_BATCH_SIZE" \
          --sample_batch_size "$RM_SAMPLE_BATCH_SIZE" \
          --max_seq_length "$RM_MAX_SEQ_LENGTH"
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

generation_args=(
  "baseline=$GEN_DIR/baseline/output_${SEED}.json"
  "htmnpo_skywork=$GEN_DIR/htmnpo_skywork/output_${SEED}.json"
  "htmnpo_athene=$GEN_DIR/htmnpo_athene/output_${SEED}.json"
)

decode_model baseline "$BASELINE_MODEL"
decode_model htmnpo_skywork "$HT_SKYWORK_MODEL"
decode_model htmnpo_athene "$HT_ATHENE_MODEL"

if [[ "$INCLUDE_ARMORM" == "1" ]]; then
  if [[ -d "$HT_ARMORM_MODEL" || -f "$GEN_DIR/htmnpo_armorm/output_${SEED}.json" ]]; then
    decode_model htmnpo_armorm "$HT_ARMORM_MODEL"
    generation_args+=("htmnpo_armorm=$GEN_DIR/htmnpo_armorm/output_${SEED}.json")
  else
    echo "Skipping htmnpo_armorm; model directory not found: $HT_ARMORM_MODEL" >&2
  fi
fi

if [[ "$INCLUDE_RONPO" == "1" ]]; then
  if [[ -d "$RONPO_MODEL" || -f "$GEN_DIR/ronpo/output_${SEED}.json" ]]; then
    decode_model ronpo "$RONPO_MODEL"
    generation_args+=("ronpo=$GEN_DIR/ronpo/output_${SEED}.json")
  else
    echo "Skipping ronpo; model directory not found: $RONPO_MODEL" >&2
  fi
fi

if [[ "$INCLUDE_SPPO" == "1" ]]; then
  if [[ -d "$SPPO_MODEL" || -f "$GEN_DIR/sppo/output_${SEED}.json" ]]; then
    decode_model sppo "$SPPO_MODEL"
    generation_args+=("sppo=$GEN_DIR/sppo/output_${SEED}.json")
  else
    echo "Skipping sppo; model directory not found: $SPPO_MODEL" >&2
  fi
fi

if [[ "$INCLUDE_INPO" == "1" ]]; then
  if [[ -d "$INPO_MODEL" || -f "$GEN_DIR/inpo/output_${SEED}.json" ]]; then
    decode_model inpo "$INPO_MODEL"
    generation_args+=("inpo=$GEN_DIR/inpo/output_${SEED}.json")
  else
    echo "Skipping inpo; model directory not found: $INPO_MODEL" >&2
  fi
fi

"$PYTHON_TRAIN" -m mnpo_scripts.merge_model_generations \
  --generations "${generation_args[@]}" \
  --output_file "$WORK_DIR/merged_model_generations_extended.json"

score_objectives "$WORK_DIR/merged_model_generations_extended.json"

# shellcheck disable=SC2046
"$PYTHON_TRAIN" -m mnpo_scripts.evaluate_multi_objective_models \
  --scored_files $(objective_args) \
  --output_dir "$RESULT_DIR" \
  --baseline_model baseline

"$PYTHON_TRAIN" -m mnpo_scripts.write_mnpo_benchmark_table \
  --result_dir "$RESULT_DIR" \
  --output_prefix "$RESULT_DIR/base_vs_htmnpo_ronpo_stage${STAGE}_extended" \
  --objectives "$OBJECTIVES"

echo "Extended evaluation written to $RESULT_DIR"
