#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=setup_ext_cache.sh
source "$PROJECT_ROOT/scripts/setup_ext_cache.sh"

export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT}"

PYTHON_INFER="${PYTHON_INFER:-/home/sjkim/anaconda3/envs/mnpo_infer/bin/python}"
PYTHON_TRAIN="${PYTHON_TRAIN:-/home/sjkim/anaconda3/envs/mnpo_train/bin/python}"

GPU="${GPU:-0}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE:-128}"
DECODE_GPU_MEMORY_UTILIZATION="${DECODE_GPU_MEMORY_UTILIZATION:-0.85}"
DECODE_ATTENTION_BACKEND="${DECODE_ATTENTION_BACKEND:-XFORMERS}"
DECODE_DTYPE="${DECODE_DTYPE:-bfloat16}"

SKYWORK_MODEL="${SKYWORK_MODEL:-Skywork/Skywork-Reward-V2-Llama-3.1-8B}"
GUARD_MODEL="${GUARD_MODEL:-Qwen/Qwen3Guard-Gen-0.6B}"
BREVITY_TARGET_WORDS="${BREVITY_TARGET_WORDS:-180}"
BREVITY_TOLERANCE_WORDS="${BREVITY_TOLERANCE_WORDS:-80}"

EXP_ROOT="${EXP_ROOT:-$PROJECT_ROOT/experiments/ronpo_safety_conflict_qwen25_1p5b_20260629}"
OUT_ROOT="${OUT_ROOT:-/ext_hdd/sjkim/mnpo/experiments/ronpo_safety_conflict_qwen25_1p5b_20260629}"
EVAL_FILE="${EVAL_FILE:-$EXP_ROOT/pairs/full_atom/test_merged_scores.jsonl}"
WORK_DIR="${WORK_DIR:-$EXP_ROOT/reward_eval_gpu0_$(date +%Y%m%d_%H%M%S)}"
GEN_DIR="$WORK_DIR/generations"
SCORED_DIR="$WORK_DIR/scored"
RESULT_DIR="$WORK_DIR/results"
LOG_DIR="$WORK_DIR/logs"

FULL_ROOT="${FULL_ROOT:-$OUT_ROOT/outputs/ronpo-safe-full-s1_seed42}"
KONLY_ROOT="${KONLY_ROOT:-$OUT_ROOT/outputs/ronpo-safe-konly-s1_seed42}"

mkdir -p "$GEN_DIR" "$SCORED_DIR" "$RESULT_DIR" "$LOG_DIR"

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required directory: $1" >&2
    exit 1
  fi
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
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_tokens "$MAX_TOKENS" \
    --batch_size "$DECODE_BATCH_SIZE" \
    --attention_backend "$DECODE_ATTENTION_BACKEND" \
    --dtype "$DECODE_DTYPE" \
    --gpu_memory_utilization "$DECODE_GPU_MEMORY_UTILIZATION" \
    --cache_dir "$HUGGINGFACE_HUB_CACHE" \
    >"$LOG_DIR/decode_${name}.log" 2>&1
}

score_skywork() {
  echo "[score:helpfulness] $SKYWORK_MODEL gpu=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_INFER" -u -m on_policy_data_gen.rm_skywork \
    --model_name "$SKYWORK_MODEL" \
    --input_file "$WORK_DIR/merged_model_generations.json" \
    --output_file "$SCORED_DIR/eval_helpfulness_skywork.jsonl" \
    --cache_dir "$HUGGINGFACE_HUB_CACHE" \
    --batch_size "${SKYWORK_BATCH_SIZE:-8}" \
    --sample_batch_size "${SKYWORK_SAMPLE_BATCH_SIZE:-16}" \
    --attn_implementation "${SKYWORK_ATTN_IMPLEMENTATION:-sdpa}" \
    >"$LOG_DIR/score_helpfulness_skywork.log" 2>&1
}

score_safety() {
  echo "[score:safety] $GUARD_MODEL gpu=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_INFER" -u -m on_policy_data_gen.rm_qwen3guard \
    --model_name "$GUARD_MODEL" \
    --cache_dir "$HF_HOME" \
    --input_file "$WORK_DIR/merged_model_generations.json" \
    --output_file "$SCORED_DIR/eval_safety_qwen3guard.jsonl" \
    --batch_size "${GUARD_BATCH_SIZE:-32}" \
    --sample_batch_size "${GUARD_SAMPLE_BATCH_SIZE:-32}" \
    --max_input_length "${GUARD_MAX_INPUT_LENGTH:-4096}" \
    --max_new_tokens "${GUARD_MAX_NEW_TOKENS:-48}" \
    --local_files_only \
    >"$LOG_DIR/score_safety_qwen3guard.log" 2>&1
}

score_brevity() {
  echo "[score:brevity] target=${BREVITY_TARGET_WORDS} tolerance=${BREVITY_TOLERANCE_WORDS}"
  "$PYTHON_TRAIN" -m mnpo_scripts.score_brevity_and_collapse \
    --input_file "$WORK_DIR/merged_model_generations.json" \
    --output_file "$SCORED_DIR/eval_brevity.jsonl" \
    --collapse_csv "$RESULT_DIR/collapse_diagnostics.csv" \
    --target_words "$BREVITY_TARGET_WORDS" \
    --tolerance_words "$BREVITY_TOLERANCE_WORDS" \
    >"$LOG_DIR/score_brevity.log" 2>&1
}

require_file "$EVAL_FILE"

echo "[start] RONPO safety conflict reward eval $(date -Is)"
echo "[work] $WORK_DIR"
echo "[gpu] $GPU"
echo "[eval_file] $EVAL_FILE"
echo "[gen] seed=$SEED temp=$TEMPERATURE top_p=$TOP_P max_tokens=$MAX_TOKENS batch=$DECODE_BATCH_SIZE"
nvidia-smi

if [[ -n "${EVAL_SPECS:-}" ]]; then
  mapfile -t merge_args < <(printf '%s\n' "$EVAL_SPECS" | sed '/^[[:space:]]*$/d')
else
  declare -a merge_args=(
    "base=Qwen/Qwen2.5-1.5B-Instruct"
    "full_s700=$FULL_ROOT/checkpoint-700"
    "full_s800=$FULL_ROOT/checkpoint-800"
    "full_s900=$FULL_ROOT/checkpoint-900"
    "konly_s800=$KONLY_ROOT/checkpoint-800"
    "konly_s900=$KONLY_ROOT/checkpoint-900"
    "konly_s1000=$KONLY_ROOT/checkpoint-1000"
  )
fi

declare -a merge_paths=()
for spec in "${merge_args[@]}"; do
  name="${spec%%=*}"
  model="${spec#*=}"
  if [[ "$model" == /* ]]; then
    require_dir "$model"
  fi
  decode_model "$name" "$model"
  merge_paths+=("$name=$GEN_DIR/$name/output_${SEED}.json")
done

"$PYTHON_TRAIN" -m mnpo_scripts.merge_model_generations \
  --generations "${merge_paths[@]}" \
  --output_file "$WORK_DIR/merged_model_generations.json" \
  >"$LOG_DIR/merge.log" 2>&1

score_skywork
score_safety
score_brevity

"$PYTHON_TRAIN" -m mnpo_scripts.evaluate_multi_objective_models \
  --scored_files \
    "helpfulness=$SCORED_DIR/eval_helpfulness_skywork.jsonl" \
    "safety=$SCORED_DIR/eval_safety_qwen3guard.jsonl" \
    "brevity=$SCORED_DIR/eval_brevity.jsonl" \
  --output_dir "$RESULT_DIR" \
  --baseline_model base \
  >"$LOG_DIR/evaluate.log" 2>&1

"$PYTHON_TRAIN" -m mnpo_scripts.write_conflict_reward_eval_report \
  --result_dir "$RESULT_DIR" \
  --collapse_csv "$RESULT_DIR/collapse_diagnostics.csv" \
  --output_md "$RESULT_DIR/report.md" \
  --title "RONPO Safety Conflict Reward Evaluation" \
  --generation_config "seed=$SEED, temperature=$TEMPERATURE, top_p=$TOP_P, max_tokens=$MAX_TOKENS, GPU=$GPU" \
  >"$LOG_DIR/write_report.log" 2>&1

echo "[done] results written to $RESULT_DIR $(date -Is)"
