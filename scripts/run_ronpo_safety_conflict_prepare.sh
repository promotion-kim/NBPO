#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$PROJECT_ROOT/scripts/setup_ext_cache.sh" ]]; then
  # shellcheck source=scripts/setup_ext_cache.sh
  source "$PROJECT_ROOT/scripts/setup_ext_cache.sh"
fi

export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT}"
export HF_HOME="${HF_HOME:-/ext_hdd/sjkim/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export WANDB_ENTITY="${WANDB_ENTITY:-promotion-kim}"
export WANDB_PROJECT="${WANDB_PROJECT:-mnpo}"

PYTHON_GUARD="${PYTHON_GUARD:-/home/sjkim/anaconda3/envs/mnpo_infer/bin/python}"
PYTHON_ARMO="${PYTHON_ARMO:-/home/sjkim/anaconda3/envs/mnpo_train/bin/python}"
PYTHON_TRAIN="${PYTHON_TRAIN:-/home/sjkim/anaconda3/envs/mnpo_train/bin/python}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
GUARD_MODEL="${GUARD_MODEL:-Qwen/Qwen3Guard-Gen-0.6B}"
SAFETY_SCORER="${SAFETY_SCORER:-qwen3guard}"
ARMO_SAFE_HEAD="${ARMO_SAFE_HEAD:-beavertails-is_safe}"

EXP_ROOT="${EXP_ROOT:-$PROJECT_ROOT/experiments/ronpo_safety_conflict_qwen25_1p5b_20260629}"
OUT_ROOT="${OUT_ROOT:-/ext_hdd/sjkim/mnpo/experiments/ronpo_safety_conflict_qwen25_1p5b_20260629}"
BASE_SCORED="${BASE_SCORED:-$PROJECT_ROOT/data/qwen2.5-1.5b-instruct_online_htmnpo_ronpo/base/iter1/scored}"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$EXP_ROOT" "$OUT_ROOT" "$LOG_DIR"

LOG_FILE="$LOG_DIR/prepare_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

SAFETY_NUM_SHARDS="${SAFETY_NUM_SHARDS:-2}"
SAFETY_GPUS="${SAFETY_GPUS:-${GPU_A:-1},${GPU_B:-2}}"
IFS=',' read -r -a SAFETY_GPU_LIST <<< "$SAFETY_GPUS"
if [[ ${#SAFETY_GPU_LIST[@]} -lt 1 ]]; then
  echo "SAFETY_GPUS must contain at least one GPU id" >&2
  exit 2
fi

echo "[start] $(date -Is) safety-conflict prepare"
echo "[models] base=$BASE_MODEL safety_scorer=$SAFETY_SCORER guard=$GUARD_MODEL armo_head=$ARMO_SAFE_HEAD"
echo "[safety-shards] num_shards=$SAFETY_NUM_SHARDS gpus=$SAFETY_GPUS"
echo "[paths] exp=$EXP_ROOT out=$OUT_ROOT log=$LOG_FILE"
nvidia-smi

score_safety_shard() {
  local split="$1"
  local shard="$2"
  local gpu="$3"
  local input="$BASE_SCORED/${split}_skywork.jsonl"
  local output="$OUT_ROOT/scored/${split}_safety_shard${shard}.jsonl"
  local done_marker="${output}.done"
  if [[ -s "$output" && -f "$done_marker" ]]; then
    echo "[guard:$split:$shard] reuse $output"
    return 0
  fi
  echo "[safety:$split:$shard] scorer=$SAFETY_SCORER gpu=$gpu input=$input output=$output"
  case "$SAFETY_SCORER" in
    qwen3guard)
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_GUARD" "$PROJECT_ROOT/on_policy_data_gen/rm_qwen3guard.py" \
        --model_name "$GUARD_MODEL" \
        --cache_dir "$CACHE_DIR" \
        --input_file "$input" \
        --output_file "$output" \
        --batch_size "${GUARD_BATCH_SIZE:-32}" \
        --sample_batch_size "${GUARD_SAMPLE_BATCH_SIZE:-32}" \
        --max_input_length "${GUARD_MAX_INPUT_LENGTH:-4096}" \
        --max_new_tokens "${GUARD_MAX_NEW_TOKENS:-48}" \
        --num_shards "$SAFETY_NUM_SHARDS" \
        --shard_index "$shard" \
        --local_files_only
      ;;
    armo_safe)
      armo_args=(
        "$PROJECT_ROOT/on_policy_data_gen/rm_armo.py"
        --cache_dir "$CACHE_DIR" \
        --input_file "$input" \
        --output_file "$output" \
        --batch_size "${ARMO_BATCH_SIZE:-8}" \
        --sample_batch_size "${ARMO_SAMPLE_BATCH_SIZE:-16}" \
        --max_seq_length "${ARMO_MAX_SEQ_LENGTH:-4096}" \
        --reward_attribute_name "$ARMO_SAFE_HEAD" \
        --local_files_only "${ARMO_LOCAL_FILES_ONLY:-true}" \
        --num_shards "$SAFETY_NUM_SHARDS" \
        --shard_index "$shard"
      )
      if [[ -n "${ARMO_MAX_SAMPLES:-}" ]]; then
        armo_args+=(--max_samples "$ARMO_MAX_SAMPLES")
      fi
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_ARMO" "${armo_args[@]}"
      ;;
    *)
      echo "Unsupported SAFETY_SCORER=$SAFETY_SCORER. Expected qwen3guard or armo_safe." >&2
      exit 2
      ;;
  esac
  touch "$done_marker"
}

merge_guard_split() {
  local split="$1"
  local merged="$OUT_ROOT/scored/${split}_safety.jsonl"
  local done_marker="${merged}.done"
  if [[ -s "$merged" && -f "$done_marker" ]]; then
  echo "[safety:$split] reuse merged $merged"
    return 0
  fi
  : > "$merged"
  for shard in $(seq 0 $((SAFETY_NUM_SHARDS - 1))); do
    cat "$OUT_ROOT/scored/${split}_safety_shard${shard}.jsonl" >> "$merged"
  done
  touch "$done_marker"
  echo "[safety:$split] merged -> $merged"
}

run_safety_split() {
  local split="$1"
  local pids=()
  for shard in $(seq 0 $((SAFETY_NUM_SHARDS - 1))); do
    local gpu_idx=$((shard % ${#SAFETY_GPU_LIST[@]}))
    local gpu="${SAFETY_GPU_LIST[$gpu_idx]}"
    score_safety_shard "$split" "$shard" "$gpu" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  merge_guard_split "$split"
}

run_safety_split train
run_safety_split test

"$PYTHON_TRAIN" "$PROJECT_ROOT/mnpo_scripts/build_safety_brevity_objectives.py" \
  --train_helpfulness "$BASE_SCORED/train_skywork.jsonl" \
  --test_helpfulness "$BASE_SCORED/test_skywork.jsonl" \
  --train_safety "$OUT_ROOT/scored/train_safety.jsonl" \
  --test_safety "$OUT_ROOT/scored/test_safety.jsonl" \
  --output_dir "$EXP_ROOT" \
  --target_words "${BREVITY_TARGET_WORDS:-180}" \
  --tolerance_words "${BREVITY_TOLERANCE_WORDS:-80}"

build_pairs() {
  local mode="$1"
  local strategy="$2"
  local pair_root="$EXP_ROOT/pairs/$mode"
  mkdir -p "$pair_root"
  for split in train test; do
    "$PYTHON_TRAIN" "$PROJECT_ROOT/mnpo_scripts/build_multi_objective_dataset.py" \
      --scored_files \
        "helpfulness=$EXP_ROOT/scored/${split}_helpfulness.jsonl" \
        "safety=$EXP_ROOT/scored/${split}_safety.jsonl" \
        "brevity=$EXP_ROOT/scored/${split}_brevity.jsonl" \
      --mnpo_output "$pair_root/${split}_mnpo_average_unused.jsonl" \
      --ronpo_output "$pair_root/${split}_ronpo.jsonl" \
      --merged_output "$pair_root/${split}_merged_scores.jsonl" \
      --summary_output "$pair_root/${split}_summary.csv" \
      --normalization minmax \
      --ronpo_pair_strategy "$strategy" \
      --adversary_steps "${ADVERSARY_STEPS:-25}" \
      --adversary_alpha "${ADVERSARY_ALPHA:-1.0}" \
      --adversary_kappa "${ADVERSARY_KAPPA:-0.05}" \
      --preference_scale "${PREFERENCE_SCALE:-8.0}" \
      --policy_mode "${POLICY_MODE:-uniform}" \
      --pairs_per_prompt "${PAIRS_PER_PROMPT:-3}" \
      --adversary_selection "${ADVERSARY_SELECTION:-all}" \
      --ronpo_policy_pair_mode "${RONPO_POLICY_PAIR_MODE:-expected_relative_policy_vs_policy}" \
      --ronpo_policy_samples_per_atom "${RONPO_POLICY_SAMPLES_PER_ATOM:-1}" \
      --k_only_fixed_atom avg_worst \
      --k_only_response_mode "${K_ONLY_RESPONSE_MODE:-uniform}" \
      --common_pair_seed "${COMMON_PAIR_SEED:-42}"
  done
}

build_pairs full_atom sigma
build_pairs k_only sigma_k_only
build_pairs uniform uniform
build_pairs a_only sigma_a_only
build_pairs maxmin_pointwise maxmin_pointwise

"$PYTHON_TRAIN" -m mnpo_scripts.analyze_conflict_gate \
  --scored_files \
    "helpfulness=$EXP_ROOT/scored/test_helpfulness.jsonl" \
    "safety=$EXP_ROOT/scored/test_safety.jsonl" \
    "concise=$EXP_ROOT/scored/test_brevity.jsonl" \
  --output_dir "$EXP_ROOT/gate1_test" \
  --title "RONPO Gate 1 Conflict Diagnostics (${SAFETY_SCORER})"

echo "[prepared] $(date -Is)"
echo "[report] $EXP_ROOT/report.md"

if [[ "${AUTO_LAUNCH:-0}" == "1" ]]; then
  tmux new-session -d -s ronpo_safety_full \
    "cd '$PROJECT_ROOT' && EXP_ROOT='$EXP_ROOT' OUT_ROOT='$OUT_ROOT' scripts/run_ronpo_safety_conflict_train.sh full_atom '${GPU_A:-1}' 42"
  tmux new-session -d -s ronpo_safety_konly \
    "cd '$PROJECT_ROOT' && EXP_ROOT='$EXP_ROOT' OUT_ROOT='$OUT_ROOT' scripts/run_ronpo_safety_conflict_train.sh k_only '${GPU_B:-2}' 42"
  echo "[launched] tmux sessions: ronpo_safety_full, ronpo_safety_konly"
else
  echo "[launch skipped] AUTO_LAUNCH=0"
fi
