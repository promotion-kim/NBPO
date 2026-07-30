#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${USE_EXT_CACHE:-1}" == "1" ]]; then
  # shellcheck source=../scripts/setup_ext_cache.sh
  source "$PROJECT_ROOT/scripts/setup_ext_cache.sh"
fi

PYTHON_TRAIN="${PYTHON_TRAIN:-/home/sjkim/anaconda3/envs/mnpo_train/bin/python}"
PYTHON_INFER="${PYTHON_INFER:-/home/sjkim/anaconda3/envs/mnpo_infer/bin/python}"
# Use reward-model-specific environments. Skywork/Athene tokenizers load under
# mnpo_infer; ArmoRM custom code loads under mnpo_train.
PYTHON_RM_SKYWORK="${PYTHON_RM_SKYWORK:-$PYTHON_INFER}"
PYTHON_RM_ATHENE="${PYTHON_RM_ATHENE:-$PYTHON_INFER}"
PYTHON_RM_ARMO="${PYTHON_RM_ARMO:-$PYTHON_TRAIN}"

STAGE="${STAGE:-1}"
EVAL_FILE="${EVAL_FILE:-$PROJECT_ROOT/data/gemma2_ufb_part1_test.jsonl}"
WORK_DIR="${WORK_DIR:-$PROJECT_ROOT/data/qwen2.5-1.5b-instruct_htmnpo_ronpo_eval_stage${STAGE}}"
GEN_DIR="${GEN_DIR:-$WORK_DIR/generations}"
SCORED_DIR="${SCORED_DIR:-$WORK_DIR/scored}"
RESULT_DIR="${RESULT_DIR:-$WORK_DIR/results}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/ext_hdd/sjkim/mnpo/outputs}"

BASELINE_MODEL="${BASELINE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
HT_SKYWORK_MODEL="${HT_SKYWORK_MODEL:-$OUTPUT_ROOT/qwen2.5-1.5b-instruct_htmnpo_skywork_online_multiobj_stage_${STAGE}}"
HT_ATHENE_MODEL="${HT_ATHENE_MODEL:-$OUTPUT_ROOT/qwen2.5-1.5b-instruct_htmnpo_athene_online_multiobj_stage_${STAGE}}"
HT_ARMO_MODEL="${HT_ARMO_MODEL:-$OUTPUT_ROOT/qwen2.5-1.5b-instruct_htmnpo_armo_online_multiobj_stage_${STAGE}}"
RONPO_MODEL="${RONPO_MODEL:-$OUTPUT_ROOT/qwen2.5-1.5b-instruct_ronpo_vs_htmnpo_online_multiobj_stage_${STAGE}}"

OBJECTIVES="${OBJECTIVES:-skywork,athene,armo}"
SEED="${SEED:-42}"
DECODE_GPUS="${DECODE_GPUS:-1}"
DECODE_ATTENTION_BACKEND="${DECODE_ATTENTION_BACKEND:-XFORMERS}"

export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

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
    "$PYTHON_INFER" -u -m on_policy_data_gen.decode \
      --data_dir "$EVAL_FILE" \
      --model "$model" \
      --seeds "$SEED" \
      --output_dir "$out_dir" \
      --num_gpu "$DECODE_GPUS" \
      --temperature 0.7 \
      --top_p 0.9 \
      "${backend_args[@]}" \
      "${cache_args[@]}"
  fi
}

score_objectives() {
  local input_json="$1"
  IFS=',' read -ra names <<< "$OBJECTIVES"
  for obj in "${names[@]}"; do
    local output_file="$SCORED_DIR/eval_${obj}.jsonl"
    if [[ "${FORCE_SCORE:-0}" != "1" && -f "$output_file" ]]; then
      continue
    fi
    case "$obj" in
      skywork)
        "$PYTHON_RM_SKYWORK" -u -m on_policy_data_gen.rm_skywork \
          --input_file "$input_json" \
          --output_file "$output_file" \
          --cache_dir "${CACHE_DIR:-/cache}"
        ;;
      athene)
        "$PYTHON_RM_ATHENE" -u -m on_policy_data_gen.rm_athene \
          --input_file "$input_json" \
          --output_file "$output_file" \
          --cache_dir "${CACHE_DIR:-/cache}"
        ;;
      armo)
        "$PYTHON_RM_ARMO" -u -m on_policy_data_gen.rm_armo \
          --input_file "$input_json" \
          --output_file "$output_file" \
          --cache_dir "${CACHE_DIR:-/cache}"
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

decode_model baseline "$BASELINE_MODEL"
decode_model htmnpo_skywork "$HT_SKYWORK_MODEL"
decode_model htmnpo_athene "$HT_ATHENE_MODEL"
decode_model htmnpo_armo "$HT_ARMO_MODEL"
decode_model ronpo "$RONPO_MODEL"

"$PYTHON_TRAIN" -m mnpo_scripts.merge_model_generations \
  --generations \
    "baseline=$GEN_DIR/baseline/output_${SEED}.json" \
    "htmnpo_skywork=$GEN_DIR/htmnpo_skywork/output_${SEED}.json" \
    "htmnpo_athene=$GEN_DIR/htmnpo_athene/output_${SEED}.json" \
    "htmnpo_armo=$GEN_DIR/htmnpo_armo/output_${SEED}.json" \
    "ronpo=$GEN_DIR/ronpo/output_${SEED}.json" \
  --output_file "$WORK_DIR/merged_model_generations.json"

score_objectives "$WORK_DIR/merged_model_generations.json"

# shellcheck disable=SC2046
"$PYTHON_TRAIN" -m mnpo_scripts.evaluate_multi_objective_models \
  --scored_files $(objective_args) \
  --output_dir "$RESULT_DIR" \
  --baseline_model baseline

echo "HT-MNPO vs RONPO robust evaluation written to $RESULT_DIR"
