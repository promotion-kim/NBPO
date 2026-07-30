#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${USE_EXT_CACHE:-1}" == "1" ]]; then
  # shellcheck source=scripts/setup_ext_cache.sh
  source "$PROJECT_ROOT/scripts/setup_ext_cache.sh"
fi

PYTHON_TRAIN="${PYTHON_TRAIN:-/home/sjkim/anaconda3/envs/mnpo_train/bin/python}"
PYTHON_INFER="${PYTHON_INFER:-/home/sjkim/anaconda3/envs/mnpo_infer/bin/python}"
# A single reward-model environment is not reliable here: Skywork/Athene need
# the inference tokenizer stack, while ArmoRM needs the training stack.
PYTHON_RM_SKYWORK="${PYTHON_RM_SKYWORK:-$PYTHON_INFER}"
PYTHON_RM_ATHENE="${PYTHON_RM_ATHENE:-$PYTHON_INFER}"
PYTHON_RM_ARMO="${PYTHON_RM_ARMO:-$PYTHON_TRAIN}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
WORK_ROOT="${WORK_ROOT:-$PROJECT_ROOT/data/qwen2.5-1.5b-instruct_online_htmnpo_ronpo}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/ext_hdd/sjkim/mnpo/outputs}"
HT_CONFIG="${HT_CONFIG:-$PROJECT_ROOT/training_configs/mnpo/qwen2.5-1.5b-instruct-ht-mnpo-multiobj-iter1.yaml}"
RONPO_CONFIG="${RONPO_CONFIG:-$PROJECT_ROOT/training_configs/ronpo/qwen2.5-1.5b-instruct-ronpo-multiobj-iter1.yaml}"
INPO_CONFIG="${INPO_CONFIG:-$PROJECT_ROOT/training_configs/inpo/qwen2.5-1.5b-instruct-inpo-avg-multiobj-iter1.yaml}"
SPPO_CONFIG="${SPPO_CONFIG:-$PROJECT_ROOT/training_configs/sppo/qwen2.5-1.5b-instruct-sppo-avg-multiobj-iter1.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$PROJECT_ROOT/accelerate_configs/deepspeed_zero3.yaml}"

STAGES="${STAGES:-1}"
PLAYERS="${PLAYERS:-skywork,athene,armo}"
HT_HISTORY_PLAYERS="${HT_HISTORY_PLAYERS:-$PLAYERS}"
RUN_HTMNPO="${RUN_HTMNPO:-1}"
RUN_RONPO="${RUN_RONPO:-1}"
RUN_INPO="${RUN_INPO:-0}"
RUN_SPPO="${RUN_SPPO:-0}"
SEEDS="${SEEDS:-13 21 42 79 100}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
DECODE_GPUS="${DECODE_GPUS:-1}"
DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE:-512}"
DECODE_ATTENTION_BACKEND="${DECODE_ATTENTION_BACKEND:-XFORMERS}"
RM_GPUS="${RM_GPUS:-}"
RM_PARALLEL="${RM_PARALLEL:-1}"
RM_BATCH_SIZE="${RM_BATCH_SIZE:-16}"
RM_SAMPLE_BATCH_SIZE="${RM_SAMPLE_BATCH_SIZE:-64}"
RM_MAX_SEQ_LENGTH="${RM_MAX_SEQ_LENGTH:-4096}"
RM_MAX_SAMPLES="${RM_MAX_SAMPLES:-}"
PRECOMPUTE_BATCH_SIZE="${PRECOMPUTE_BATCH_SIZE:-2}"
PRECOMPUTE_MAX_LENGTH="${PRECOMPUTE_MAX_LENGTH:-2048}"
PRECOMPUTE_MAX_PROMPT_LENGTH="${PRECOMPUTE_MAX_PROMPT_LENGTH:-1800}"
PRECOMPUTE_APPLY_CHAT_TEMPLATE="${PRECOMPUTE_APPLY_CHAT_TEMPLATE:-true}"
PRECOMPUTE_AUTO_INSERT_EMPTY_SYSTEM_MSG="${PRECOMPUTE_AUTO_INSERT_EMPTY_SYSTEM_MSG:-false}"
TRAIN_PER_DEVICE_BATCH_SIZE="${TRAIN_PER_DEVICE_BATCH_SIZE:-2}"
TRAIN_PER_DEVICE_EVAL_BATCH_SIZE="${TRAIN_PER_DEVICE_EVAL_BATCH_SIZE:-2}"
TRAIN_GRADIENT_ACCUMULATION_STEPS="${TRAIN_GRADIENT_ACCUMULATION_STEPS:-8}"
TRAIN_MAX_LENGTH="${TRAIN_MAX_LENGTH:-2048}"
TRAIN_MAX_PROMPT_LENGTH="${TRAIN_MAX_PROMPT_LENGTH:-1800}"
TRAIN_GENERATE_DURING_EVAL="${TRAIN_GENERATE_DURING_EVAL:-true}"
TRAIN_EVAL_GENERATION_SAMPLES="${TRAIN_EVAL_GENERATION_SAMPLES:-5}"
TRAIN_EVAL_GENERATION_MAX_NEW_TOKENS="${TRAIN_EVAL_GENERATION_MAX_NEW_TOKENS:-256}"
TRAIN_EVAL_GENERATION_DO_SAMPLE="${TRAIN_EVAL_GENERATION_DO_SAMPLE:-false}"
TRAIN_EVAL_GENERATION_TEMPERATURE="${TRAIN_EVAL_GENERATION_TEMPERATURE:-0.7}"
TRAIN_EVAL_GENERATION_TOP_P="${TRAIN_EVAL_GENERATION_TOP_P:-0.9}"
TRAIN_EVAL_GENERATION_TOP_K="${TRAIN_EVAL_GENERATION_TOP_K:-20}"
TRAIN_EVAL_GENERATION_BACKEND="${TRAIN_EVAL_GENERATION_BACKEND:-checkpoint}"
TRAIN_EVAL_GENERATION_OUTPUT_DIR="${TRAIN_EVAL_GENERATION_OUTPUT_DIR:-/ext_hdd/sjkim/mnpo/eval_generations}"
TRAIN_EVAL_GENERATION_DEVICE="${TRAIN_EVAL_GENERATION_DEVICE:-cuda}"
TRAIN_EVAL_GENERATION_CUDA_VISIBLE_DEVICES="${TRAIN_EVAL_GENERATION_CUDA_VISIBLE_DEVICES:-}"
TRAIN_EVAL_GENERATION_DTYPE="${TRAIN_EVAL_GENERATION_DTYPE:-bfloat16}"
TRAIN_EVAL_GENERATION_KEEP_SNAPSHOT="${TRAIN_EVAL_GENERATION_KEEP_SNAPSHOT:-false}"
TRAIN_EVAL_GENERATION_LOCAL_FILES_ONLY="${TRAIN_EVAL_GENERATION_LOCAL_FILES_ONLY:-true}"
TRAIN_EVAL_GENERATION_PRINT_MAX_CHARS="${TRAIN_EVAL_GENERATION_PRINT_MAX_CHARS:-1200}"
TRAIN_EVAL_STEPS="${TRAIN_EVAL_STEPS:-100}"
TRAIN_SAVE_STEPS="${TRAIN_SAVE_STEPS:-100}"
TRAIN_SAVE_TOTAL_LIMIT="${TRAIN_SAVE_TOTAL_LIMIT:-5}"
TRAIN_LOGGING_STEPS="${TRAIN_LOGGING_STEPS:-5}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-}"
TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-}"
TRAIN_WARMUP_RATIO="${TRAIN_WARMUP_RATIO:-}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-}"

ADVERSARY_STEPS="${ADVERSARY_STEPS:-25}"
ADVERSARY_ALPHA="${ADVERSARY_ALPHA:-1.0}"
ADVERSARY_KAPPA="${ADVERSARY_KAPPA:-0.05}"
PREFERENCE_SCALE="${PREFERENCE_SCALE:-8.0}"
PAIRS_PER_PROMPT="${PAIRS_PER_PROMPT:-1}"
RONPO_POLICY_PAIR_MODE="${RONPO_POLICY_PAIR_MODE:-best_vs_adversary}"
RONPO_POLICY_SAMPLES_PER_ATOM="${RONPO_POLICY_SAMPLES_PER_ATOM:-0}"
RONPO_CACHE_TAG="${RONPO_CACHE_TAG:-sigma_${RONPO_POLICY_PAIR_MODE}_pairs${PAIRS_PER_PROMPT}_samples${RONPO_POLICY_SAMPLES_PER_ATOM}}"
HT_TARGET_MODE="${HT_TARGET_MODE:-reward_gap}"
HT_NORMALIZATION="${HT_NORMALIZATION:-none}"
SHARE_STAGE1_BASE_DATA="${SHARE_STAGE1_BASE_DATA:-1}"
BASELINE_REUSE_RONPO_DATA="${BASELINE_REUSE_RONPO_DATA:-1}"
BASELINE_WAIT_FOR_SHARED_DATA="${BASELINE_WAIT_FOR_SHARED_DATA:-0}"
BASELINE_WAIT_TIMEOUT_SECONDS="${BASELINE_WAIT_TIMEOUT_SECONDS:-43200}"
BASELINE_STAGE1_POLICY="${BASELINE_STAGE1_POLICY:-}"
INPO_ETA="${INPO_ETA:-0.0075}"
INPO_RATIO="${INPO_RATIO:-0.3333}"
INPO_BETA="${INPO_BETA:-10}"
SPPO_ETA="${SPPO_ETA:-0.0075}"
SPPO_RATIO="${SPPO_RATIO:-0.3333}"
SPPO_BETA="${SPPO_BETA:-10}"

export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp3s0f0}"
export WANDB_ENTITY="${WANDB_ENTITY:-promotion-kim}"
export WANDB_PROJECT="${WANDB_PROJECT:-mnpo}"

if [[ -z "$RM_GPUS" ]]; then
  RM_GPUS="$CUDA_VISIBLE_DEVICES"
fi

mkdir -p "$WORK_ROOT" "$OUTPUT_ROOT"

IFS=',' read -ra PLAYER_LIST <<< "$PLAYERS"
IFS=',' read -ra HT_HISTORY_PLAYER_LIST <<< "$HT_HISTORY_PLAYERS"

ht_output_path() {
  local player="$1"
  local stage="$2"
  printf '%s/qwen2.5-1.5b-instruct_htmnpo_%s_online_multiobj_stage_%s' "$OUTPUT_ROOT" "$player" "$stage"
}

ronpo_output_path() {
  local stage="$1"
  printf '%s/qwen2.5-1.5b-instruct_ronpo_vs_htmnpo_online_multiobj_stage_%s' "$OUTPUT_ROOT" "$stage"
}

inpo_output_path() {
  local stage="$1"
  printf '%s/qwen2.5-1.5b-instruct_inpo_avg_online_multiobj_stage_%s' "$OUTPUT_ROOT" "$stage"
}

sppo_output_path() {
  local stage="$1"
  printf '%s/qwen2.5-1.5b-instruct_sppo_avg_online_multiobj_stage_%s' "$OUTPUT_ROOT" "$stage"
}

ht_policy_for_stage() {
  local player="$1"
  local stage="$2"
  if [[ "$stage" -eq 1 ]]; then
    printf '%s' "$BASE_MODEL"
  else
    ht_output_path "$player" "$((stage - 1))"
  fi
}

ht_ratio_for_stage() {
  local stage="$1"
  local default_ratio
  if [[ "$stage" -eq 2 ]]; then
    default_ratio="0.85"
  else
    default_ratio="0.3333"
  fi
  local var="HT_RATIO_STAGE_${stage}"
  printf '%s' "${!var:-$default_ratio}"
}

ht_eta_for_player_stage() {
  local player="$1"
  local stage="$2"
  local upper_player="${player^^}"
  local var="HT_ETA_${upper_player}_STAGE_${stage}"
  printf '%s' "${!var:-0.0075}"
}

ht_beta_for_player_stage() {
  local player="$1"
  local stage="$2"
  local default_beta
  case "${stage}:${player}" in
    1:athene) default_beta="1" ;;
    1:skywork|1:armo) default_beta="10" ;;
    2:skywork) default_beta="3" ;;
    2:athene) default_beta="1" ;;
    2:armo) default_beta="10" ;;
    *) default_beta="10" ;;
  esac
  local upper_player="${player^^}"
  local var="HT_BETA_${upper_player}_STAGE_${stage}"
  printf '%s' "${!var:-$default_beta}"
}

ronpo_policy_for_stage() {
  local stage="$1"
  if [[ "$stage" -eq 1 ]]; then
    printf '%s' "$BASE_MODEL"
  else
    ronpo_output_path "$((stage - 1))"
  fi
}

baseline_policy_for_stage() {
  local stage="$1"
  if [[ "$stage" -eq 1 ]]; then
    printf '%s' "$BASE_MODEL"
  elif [[ -n "$BASELINE_STAGE1_POLICY" ]]; then
    printf '%s' "$BASELINE_STAGE1_POLICY"
  else
    ronpo_output_path "$((stage - 1))"
  fi
}

join_by_comma() {
  local IFS=,
  echo "$*"
}

csv_contains() {
  local csv="$1"
  local needle="$2"
  local item
  IFS=',' read -ra items <<< "$csv"
  for item in "${items[@]}"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

should_force_score() {
  local obj="$1"
  local split="${2:-}"
  if [[ "${FORCE_SCORE:-0}" == "1" ]]; then
    return 0
  fi
  if [[ -n "${FORCE_SCORE_OBJECTIVES:-}" ]] && csv_contains "$FORCE_SCORE_OBJECTIVES" "$obj"; then
    if [[ -z "${FORCE_SCORE_SPLITS:-}" || -z "$split" ]] || csv_contains "$FORCE_SCORE_SPLITS" "$split"; then
      return 0
    fi
  fi
  return 1
}

objective_args_for_split() {
  local scored_dir="$1"
  local split="$2"
  local args=()
  local obj
  for obj in "${PLAYER_LIST[@]}"; do
    args+=("${obj}=${scored_dir}/${split}_${obj}.jsonl")
  done
  printf '%q ' "${args[@]}"
}

decode_split() {
  local policy_model="$1"
  local input_file="$2"
  local out_dir="$3"
  if [[ "${FORCE_DECODE:-0}" == "1" || ! -f "$out_dir/all_outputs.json" ]]; then
    local cache_args=()
    if [[ -n "${CACHE_DIR:-}" ]]; then
      cache_args=(--cache_dir "$CACHE_DIR")
    fi
    local backend_args=()
    if [[ -n "$DECODE_ATTENTION_BACKEND" ]]; then
      backend_args=(--attention_backend "$DECODE_ATTENTION_BACKEND")
    fi
    "$PYTHON_INFER" -u -m on_policy_data_gen.decode \
      --data_dir "$input_file" \
      --model "$policy_model" \
      --seeds $SEEDS \
      --output_dir "$out_dir" \
      --num_gpu "$DECODE_GPUS" \
      --batch_size "$DECODE_BATCH_SIZE" \
      "${backend_args[@]}" \
      "${cache_args[@]}"
    "$PYTHON_INFER" -m on_policy_data_gen.post_process \
      --generation_file_dir "$out_dir"
  fi
}

run_reward_scorer() {
  local obj="$1"
  local input_json="$2"
  local output_file="$3"
  shift 3
  local extra_args=("$@")
  local cache_args=()
  if [[ -n "${CACHE_DIR:-}" ]]; then
    cache_args=(--cache_dir "$CACHE_DIR")
  fi
  case "$obj" in
    skywork)
      "$PYTHON_RM_SKYWORK" -u -m on_policy_data_gen.rm_skywork \
        --input_file "$input_json" \
        --output_file "$output_file" \
        --batch_size "$RM_BATCH_SIZE" \
        --sample_batch_size "$RM_SAMPLE_BATCH_SIZE" \
        --max_seq_length "$RM_MAX_SEQ_LENGTH" \
        "${extra_args[@]}" \
        "${cache_args[@]}"
      ;;
    athene)
      "$PYTHON_RM_ATHENE" -u -m on_policy_data_gen.rm_athene \
        --input_file "$input_json" \
        --output_file "$output_file" \
        --batch_size "$RM_BATCH_SIZE" \
        --sample_batch_size "$RM_SAMPLE_BATCH_SIZE" \
        "${extra_args[@]}" \
        "${cache_args[@]}"
      ;;
    armo)
      "$PYTHON_RM_ARMO" -u -m on_policy_data_gen.rm_armo \
        --input_file "$input_json" \
        --output_file "$output_file" \
        --batch_size "$RM_BATCH_SIZE" \
        --sample_batch_size "$RM_SAMPLE_BATCH_SIZE" \
        --max_seq_length "$RM_MAX_SEQ_LENGTH" \
        "${extra_args[@]}" \
        "${cache_args[@]}"
      ;;
    *)
      echo "Unknown objective/player '$obj'. Supported: skywork,athene,armo" >&2
      exit 1
      ;;
  esac
}

score_objective_parallel() {
  local obj="$1"
  local input_json="$2"
  local output_file="$3"
  local rm_gpus_csv="${RM_GPUS:-$CUDA_VISIBLE_DEVICES}"
  local rm_gpus=()
  IFS=',' read -ra rm_gpus <<< "$rm_gpus_csv"

  if [[ "${#rm_gpus[@]}" -le 1 || "$RM_PARALLEL" != "1" ]]; then
    local max_sample_args=()
    if [[ -n "$RM_MAX_SAMPLES" ]]; then
      max_sample_args=(--max_samples "$RM_MAX_SAMPLES")
    fi
    run_reward_scorer "$obj" "$input_json" "$output_file" "${max_sample_args[@]}"
    return
  fi

  local shard_dir="${output_file}.shards"
  mkdir -p "$shard_dir"

  local shard_args=(
    --input_file "$input_json"
    --output_dir "$shard_dir"
    --num_shards "${#rm_gpus[@]}"
    --prefix input
  )
  if [[ -n "$RM_MAX_SAMPLES" ]]; then
    shard_args+=(--max_samples "$RM_MAX_SAMPLES")
  fi
  "$PYTHON_TRAIN" -m on_policy_data_gen.shard_outputs "${shard_args[@]}"

  local pids=()
  local idx
  for idx in "${!rm_gpus[@]}"; do
    local gpu="${rm_gpus[$idx]}"
    local shard_input="$shard_dir/input_${idx}.jsonl"
    local shard_output="$shard_dir/scored_${idx}.jsonl"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      run_reward_scorer "$obj" "$shard_input" "$shard_output"
    ) &
    pids+=("$!")
  done

  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "Reward scoring failed for objective '$obj'" >&2
    exit 1
  fi

  : > "$output_file"
  for idx in "${!rm_gpus[@]}"; do
    cat "$shard_dir/scored_${idx}.jsonl" >> "$output_file"
  done
}

score_objective() {
  local obj="$1"
  local input_json="$2"
  local output_file="$3"
  local split="${4:-}"
  if ! should_force_score "$obj" "$split" && [[ -f "$output_file" ]]; then
    return
  fi
  score_objective_parallel "$obj" "$input_json" "$output_file"
}

score_all_objectives() {
  local input_json="$1"
  local scored_dir="$2"
  local split="$3"
  local obj
  for obj in "${PLAYER_LIST[@]}"; do
    score_objective "$obj" "$input_json" "$scored_dir/${split}_${obj}.jsonl" "$split"
  done
}

build_ht_pairs() {
  local player="$1"
  local pair_dir="$2"
  local scored_dir="$3"
  local split="$4"
  local output_file="$pair_dir/${split}_ht_${player}.jsonl"
  if [[ "${FORCE_BUILD_PAIRS:-0}" == "1" || ! -f "$output_file" ]]; then
    "$PYTHON_TRAIN" -m mnpo_scripts.build_ht_mnpo_dataset \
      --scored_file "${player}=${scored_dir}/${split}_${player}.jsonl" \
      --output "$output_file" \
      --normalization "$HT_NORMALIZATION" \
      --target_mode "$HT_TARGET_MODE"
  fi
}

build_ronpo_pairs() {
  local pair_dir="$1"
  local scored_dir="$2"
  local split="$3"
  if [[ "${FORCE_BUILD_PAIRS:-0}" == "1" || ! -f "$pair_dir/${split}_ronpo.jsonl" ]]; then
    # shellcheck disable=SC2046
    "$PYTHON_TRAIN" -m mnpo_scripts.build_multi_objective_dataset \
      --scored_files $(objective_args_for_split "$scored_dir" "$split") \
      --mnpo_output "$pair_dir/${split}_mnpo_average_unused.jsonl" \
      --ronpo_output "$pair_dir/${split}_ronpo.jsonl" \
      --merged_output "$pair_dir/${split}_merged_scores.jsonl" \
      --summary_output "$pair_dir/${split}_ronpo_sigma_summary.csv" \
      --normalization minmax \
      --ronpo_pair_strategy sigma \
      --ronpo_policy_pair_mode "$RONPO_POLICY_PAIR_MODE" \
      --ronpo_policy_samples_per_atom "$RONPO_POLICY_SAMPLES_PER_ATOM" \
      --adversary_steps "$ADVERSARY_STEPS" \
      --adversary_alpha "$ADVERSARY_ALPHA" \
      --adversary_kappa "$ADVERSARY_KAPPA" \
      --preference_scale "$PREFERENCE_SCALE" \
      --pairs_per_prompt "$PAIRS_PER_PROMPT"
  fi
}

wait_for_path() {
  local path="$1"
  local label="${2:-$path}"
  if [[ -e "$path" ]]; then
    return
  fi
  if [[ "$BASELINE_WAIT_FOR_SHARED_DATA" != "1" ]]; then
    echo "Missing shared baseline input: ${label}. Set BASELINE_WAIT_FOR_SHARED_DATA=1 to wait, or BASELINE_REUSE_RONPO_DATA=0 to generate an independent baseline pool." >&2
    exit 1
  fi

  local start
  start="$(date +%s)"
  while [[ ! -e "$path" ]]; do
    local now
    local elapsed
    now="$(date +%s)"
    elapsed=$((now - start))
    if (( elapsed > BASELINE_WAIT_TIMEOUT_SECONDS )); then
      echo "Timed out waiting for shared baseline input after ${elapsed}s: ${label}" >&2
      exit 1
    fi
    echo "[wait] ${label} not ready after ${elapsed}s; sleeping 60s"
    sleep 60
  done
}

ensure_shared_scored_data_ready() {
  local scored_dir="$1"
  local split="$2"
  local obj
  for obj in "${PLAYER_LIST[@]}"; do
    wait_for_path "$scored_dir/${split}_${obj}.jsonl" "$scored_dir/${split}_${obj}.jsonl"
  done
}

build_avg_oracle_pairs() {
  local pair_dir="$1"
  local scored_dir="$2"
  local split="$3"
  local output_file="$pair_dir/${split}_avg_oracle.jsonl"
  if [[ "${FORCE_BUILD_PAIRS:-0}" == "1" || ! -f "$output_file" ]]; then
    # shellcheck disable=SC2046
    "$PYTHON_TRAIN" -m mnpo_scripts.build_multi_objective_dataset \
      --scored_files $(objective_args_for_split "$scored_dir" "$split") \
      --mnpo_output "$output_file" \
      --ronpo_output "$pair_dir/${split}_ronpo_unused.jsonl" \
      --merged_output "$pair_dir/${split}_merged_scores.jsonl" \
      --normalization minmax \
      --ronpo_pair_strategy none \
      --preference_scale "$PREFERENCE_SCALE" \
      --pairs_per_prompt 1
  fi
}

ht_opponent_paths_for_stage() {
  local player="$1"
  local stage="$2"
  local -n out_ref="$3"
  out_ref=()
  local override_var="HT_HISTORY_PATHS_STAGE_${stage}"
  local override_paths="${!override_var:-}"
  if [[ -n "$override_paths" ]]; then
    IFS=',' read -ra out_ref <<< "$override_paths"
    return
  fi
  local obj
  if [[ "$stage" -eq 1 ]]; then
    # All players are initialized from the same base policy; duplicate base
    # opponents collapse to the same log-ratio, so one path is sufficient.
    out_ref=("$BASE_MODEL")
    return
  fi
  for obj in "${HT_HISTORY_PLAYER_LIST[@]}"; do
    if [[ "$obj" != "$player" ]]; then
      out_ref+=("$(ht_output_path "$obj" "$((stage - 1))")")
    fi
  done
  if [[ "${#out_ref[@]}" -eq 0 ]]; then
    echo "No HT-MNPO history paths for player=${player} stage=${stage}. Set HT_HISTORY_PLAYERS or HT_HISTORY_PATHS_STAGE_${stage}." >&2
    exit 1
  fi
}

precompute_dataset() {
  local policy_model="$1"
  local pref_dir="$2"
  local train_pairs="$3"
  local test_pairs="$4"
  shift 4
  local history_paths=("$@")

  if [[ "${FORCE_PRECOMPUTE:-0}" == "1" || ! -d "$pref_dir" ]]; then
    "$PYTHON_TRAIN" -m accelerate.commands.launch --num_processes="$NUM_PROCESSES" -m mnpo_scripts.precompute \
      --model_name_or_path "$policy_model" \
      --ref_model "$BASE_MODEL" \
      --history_paths "${history_paths[@]}" \
      --train_dir "$train_pairs" \
      --test_dir "$test_pairs" \
      --output_dir "$pref_dir" \
      --per_device_train_batch_size "$PRECOMPUTE_BATCH_SIZE" \
      --max_length "$PRECOMPUTE_MAX_LENGTH" \
      --max_prompt_length "$PRECOMPUTE_MAX_PROMPT_LENGTH" \
      --apply_chat_template "$PRECOMPUTE_APPLY_CHAT_TEMPLATE" \
      --auto_insert_empty_system_msg "$PRECOMPUTE_AUTO_INSERT_EMPTY_SYSTEM_MSG" \
      --ronpo_target_mode none \
      --sanity_check False
  fi
}

train_model() {
  local config="$1"
  local policy_model="$2"
  local pref_dir="$3"
  local output_dir="$4"
  local run_name="$5"
  local history_count="$6"
  shift 6
  local extra_args=("$@")
  local weights=()
  local i
  for ((i = 0; i < history_count; i++)); do
    weights+=("1.0")
  done
  local history_weights
  history_weights="$(join_by_comma "${weights[@]}")"
  local max_steps_args=()
  if [[ -n "$TRAIN_MAX_STEPS" ]]; then
    max_steps_args=(--max_steps="$TRAIN_MAX_STEPS")
  fi
  local hparam_args=()
  if [[ -n "${TRAIN_COMMON_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    hparam_args+=($TRAIN_COMMON_EXTRA_ARGS)
  fi
  if [[ -n "$TRAIN_LEARNING_RATE" ]]; then
    hparam_args+=(--learning_rate="$TRAIN_LEARNING_RATE")
  fi
  if [[ -n "$TRAIN_WARMUP_RATIO" ]]; then
    hparam_args+=(--warmup_ratio="$TRAIN_WARMUP_RATIO")
  fi
  if [[ -n "$TRAIN_WEIGHT_DECAY" ]]; then
    hparam_args+=(--weight_decay="$TRAIN_WEIGHT_DECAY")
  fi
  local eval_gen_args=(
    --eval_generation_temperature="$TRAIN_EVAL_GENERATION_TEMPERATURE"
    --eval_generation_top_p="$TRAIN_EVAL_GENERATION_TOP_P"
    --eval_generation_top_k="$TRAIN_EVAL_GENERATION_TOP_K"
    --eval_generation_backend="$TRAIN_EVAL_GENERATION_BACKEND"
    --eval_generation_output_dir="$TRAIN_EVAL_GENERATION_OUTPUT_DIR"
    --eval_generation_device="$TRAIN_EVAL_GENERATION_DEVICE"
    --eval_generation_dtype="$TRAIN_EVAL_GENERATION_DTYPE"
    --eval_generation_keep_snapshot="$TRAIN_EVAL_GENERATION_KEEP_SNAPSHOT"
    --eval_generation_local_files_only="$TRAIN_EVAL_GENERATION_LOCAL_FILES_ONLY"
    --eval_generation_print_max_chars="$TRAIN_EVAL_GENERATION_PRINT_MAX_CHARS"
  )
  if [[ -n "$TRAIN_EVAL_GENERATION_CUDA_VISIBLE_DEVICES" ]]; then
    eval_gen_args+=(--eval_generation_cuda_visible_devices="$TRAIN_EVAL_GENERATION_CUDA_VISIBLE_DEVICES")
  fi

  ACCELERATE_LOG_LEVEL=info "$PYTHON_TRAIN" -m accelerate.commands.launch \
    --config_file "$ACCELERATE_CONFIG" \
    --num_processes="$NUM_PROCESSES" \
    -m mnpo_scripts.run_mnpo \
    "$config" \
    --model_name_or_path="$policy_model" \
    --dataset_mixer="$pref_dir:1.0" \
    --output_dir="$output_dir" \
    --run_name="$run_name" \
    --max_history_t="$history_count" \
    --history_weights="$history_weights" \
    --per_device_train_batch_size="$TRAIN_PER_DEVICE_BATCH_SIZE" \
    --per_device_eval_batch_size="$TRAIN_PER_DEVICE_EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps="$TRAIN_GRADIENT_ACCUMULATION_STEPS" \
    --max_length="$TRAIN_MAX_LENGTH" \
    --max_prompt_length="$TRAIN_MAX_PROMPT_LENGTH" \
    --generate_during_eval="$TRAIN_GENERATE_DURING_EVAL" \
    --eval_generation_samples="$TRAIN_EVAL_GENERATION_SAMPLES" \
    --eval_generation_max_new_tokens="$TRAIN_EVAL_GENERATION_MAX_NEW_TOKENS" \
    --eval_generation_do_sample="$TRAIN_EVAL_GENERATION_DO_SAMPLE" \
    "${eval_gen_args[@]}" \
    --eval_steps="$TRAIN_EVAL_STEPS" \
    --save_steps="$TRAIN_SAVE_STEPS" \
    --save_total_limit="$TRAIN_SAVE_TOTAL_LIMIT" \
    --logging_steps="$TRAIN_LOGGING_STEPS" \
    "${max_steps_args[@]}" \
    "${hparam_args[@]}" \
    "${extra_args[@]}"
}

prepare_base_stage1_data() {
  local train_file="$PROJECT_ROOT/data/gemma2_ufb_part1_train.jsonl"
  local test_file="$PROJECT_ROOT/data/gemma2_ufb_part1_test.jsonl"
  local stage_dir="$WORK_ROOT/base/iter1"
  local gen_dir="$stage_dir/generated"
  local scored_dir="$stage_dir/scored"
  mkdir -p "$gen_dir" "$scored_dir"

  decode_split "$BASE_MODEL" "$train_file" "$gen_dir/train"
  decode_split "$BASE_MODEL" "$test_file" "$gen_dir/test"
  score_all_objectives "$gen_dir/train/all_outputs.json" "$scored_dir" train
  score_all_objectives "$gen_dir/test/all_outputs.json" "$scored_dir" test
}

run_ht_player_stage() {
  local player="$1"
  local stage="$2"
  local train_file="$PROJECT_ROOT/data/gemma2_ufb_part${stage}_train.jsonl"
  local test_file="$PROJECT_ROOT/data/gemma2_ufb_part${stage}_test.jsonl"
  local policy_model
  policy_model="$(ht_policy_for_stage "$player" "$stage")"

  local stage_dir="$WORK_ROOT/htmnpo/${player}/iter${stage}"
  local gen_dir="$stage_dir/generated"
  local scored_dir="$stage_dir/scored"
  if [[ "$SHARE_STAGE1_BASE_DATA" == "1" && "$stage" -eq 1 ]]; then
    gen_dir="$WORK_ROOT/base/iter1/generated"
    scored_dir="$WORK_ROOT/base/iter1/scored"
  fi
  local pair_dir="$stage_dir/pairs"
  local pref_dir="$stage_dir/precomputed"
  mkdir -p "$gen_dir" "$scored_dir" "$pair_dir" "$stage_dir"

  echo "=== HT-MNPO player=${player} stage=${stage}: policy=${policy_model} ==="

  if ! [[ "$SHARE_STAGE1_BASE_DATA" == "1" && "$stage" -eq 1 ]]; then
    decode_split "$policy_model" "$train_file" "$gen_dir/train"
    decode_split "$policy_model" "$test_file" "$gen_dir/test"
    score_objective "$player" "$gen_dir/train/all_outputs.json" "$scored_dir/train_${player}.jsonl" train
    score_objective "$player" "$gen_dir/test/all_outputs.json" "$scored_dir/test_${player}.jsonl" test
  fi

  build_ht_pairs "$player" "$pair_dir" "$scored_dir" train
  build_ht_pairs "$player" "$pair_dir" "$scored_dir" test

  local opponents=()
  ht_opponent_paths_for_stage "$player" "$stage" opponents
  local ht_eta
  local ht_beta
  local ht_ratio
  ht_eta="$(ht_eta_for_player_stage "$player" "$stage")"
  ht_beta="$(ht_beta_for_player_stage "$player" "$stage")"
  ht_ratio="$(ht_ratio_for_stage "$stage")"
  precompute_dataset \
    "$policy_model" \
    "$pref_dir" \
    "$pair_dir/train_ht_${player}.jsonl" \
    "$pair_dir/test_ht_${player}.jsonl" \
    "${opponents[@]}"

  train_model \
    "$HT_CONFIG" \
    "$policy_model" \
    "$pref_dir" \
    "$(ht_output_path "$player" "$stage")" \
    "qwen2.5-1.5b-instruct_htmnpo_${player}_online_multiobj_stage_${stage}" \
    "${#opponents[@]}" \
    --loss_type=ht_mnpo \
    --eta="$ht_eta" \
    --ratio="$ht_ratio" \
    --beta="$ht_beta"
}

run_ronpo_stage() {
  local stage="$1"
  local train_file="$PROJECT_ROOT/data/gemma2_ufb_part${stage}_train.jsonl"
  local test_file="$PROJECT_ROOT/data/gemma2_ufb_part${stage}_test.jsonl"
  local policy_model
  policy_model="$(ronpo_policy_for_stage "$stage")"

  local stage_dir="$WORK_ROOT/ronpo/iter${stage}"
  local gen_dir="$stage_dir/generated"
  local scored_dir="$stage_dir/scored"
  if [[ "$SHARE_STAGE1_BASE_DATA" == "1" && "$stage" -eq 1 ]]; then
    gen_dir="$WORK_ROOT/base/iter1/generated"
    scored_dir="$WORK_ROOT/base/iter1/scored"
  fi
  local pair_dir="$stage_dir/pairs_${RONPO_CACHE_TAG}"
  local pref_dir="$stage_dir/precomputed_${RONPO_CACHE_TAG}"
  mkdir -p "$gen_dir" "$scored_dir" "$pair_dir" "$stage_dir"

  echo "=== RONPO stage=${stage}: policy=${policy_model} ==="

  if ! [[ "$SHARE_STAGE1_BASE_DATA" == "1" && "$stage" -eq 1 ]]; then
    decode_split "$policy_model" "$train_file" "$gen_dir/train"
    decode_split "$policy_model" "$test_file" "$gen_dir/test"
    score_all_objectives "$gen_dir/train/all_outputs.json" "$scored_dir" train
    score_all_objectives "$gen_dir/test/all_outputs.json" "$scored_dir" test
  fi

  build_ronpo_pairs "$pair_dir" "$scored_dir" train
  build_ronpo_pairs "$pair_dir" "$scored_dir" test

  precompute_dataset \
    "$policy_model" \
    "$pref_dir" \
    "$pair_dir/train_ronpo.jsonl" \
    "$pair_dir/test_ronpo.jsonl" \
    "$policy_model"

  train_model \
    "$RONPO_CONFIG" \
    "$policy_model" \
    "$pref_dir" \
    "$(ronpo_output_path "$stage")" \
    "qwen2.5-1.5b-instruct_ronpo_vs_htmnpo_online_multiobj_stage_${stage}" \
    1
}

run_avg_baseline_stage() {
  local method="$1"
  local stage="$2"
  local train_file="$PROJECT_ROOT/data/gemma2_ufb_part${stage}_train.jsonl"
  local test_file="$PROJECT_ROOT/data/gemma2_ufb_part${stage}_test.jsonl"
  local policy_model
  policy_model="$(baseline_policy_for_stage "$stage")"

  local stage_dir="$WORK_ROOT/baselines/${method}_avg/iter${stage}"
  if [[ "${BASELINE_COMPACT_NAMES:-0}" == "1" ]]; then
    stage_dir="$WORK_ROOT/${method}_s${stage}"
  fi
  local gen_dir="$stage_dir/generated"
  local scored_dir="$stage_dir/scored"
  if [[ "$BASELINE_REUSE_RONPO_DATA" == "1" ]]; then
    gen_dir="$WORK_ROOT/ronpo/iter${stage}/generated"
    scored_dir="$WORK_ROOT/ronpo/iter${stage}/scored"
    if [[ "$SHARE_STAGE1_BASE_DATA" == "1" && "$stage" -eq 1 ]]; then
      gen_dir="$WORK_ROOT/base/iter1/generated"
      scored_dir="$WORK_ROOT/base/iter1/scored"
    fi
  fi
  local pair_dir="$stage_dir/pairs_avg_oracle"
  local pref_dir="$stage_dir/precomputed_avg_oracle"
  if [[ "${BASELINE_COMPACT_NAMES:-0}" == "1" ]]; then
    pair_dir="$stage_dir/pairs"
    pref_dir="$stage_dir/precomp"
  fi
  mkdir -p "$gen_dir" "$scored_dir" "$pair_dir" "$stage_dir"

  local method_upper="${method^^}"
  echo "=== ${method_upper}-avg stage=${stage}: policy=${policy_model} ==="
  echo "=== ${method_upper}-avg homogeneous oracle: average minmax-normalized objectives over ${PLAYERS} ==="

  if [[ "$BASELINE_REUSE_RONPO_DATA" == "1" ]]; then
    ensure_shared_scored_data_ready "$scored_dir" train
    ensure_shared_scored_data_ready "$scored_dir" test
  else
    decode_split "$policy_model" "$train_file" "$gen_dir/train"
    decode_split "$policy_model" "$test_file" "$gen_dir/test"
    score_all_objectives "$gen_dir/train/all_outputs.json" "$scored_dir" train
    score_all_objectives "$gen_dir/test/all_outputs.json" "$scored_dir" test
  fi

  build_avg_oracle_pairs "$pair_dir" "$scored_dir" train
  build_avg_oracle_pairs "$pair_dir" "$scored_dir" test

  precompute_dataset \
    "$policy_model" \
    "$pref_dir" \
    "$pair_dir/train_avg_oracle.jsonl" \
    "$pair_dir/test_avg_oracle.jsonl" \
    "$policy_model"

  local output_dir
  local run_name
  case "$method" in
    inpo)
      output_dir="$(inpo_output_path "$stage")"
      run_name="qwen2.5-1.5b-instruct_inpo_avg_online_multiobj_stage_${stage}"
      ;;
    sppo)
      output_dir="$(sppo_output_path "$stage")"
      run_name="qwen2.5-1.5b-instruct_sppo_avg_online_multiobj_stage_${stage}"
      ;;
  esac
  if [[ "${BASELINE_COMPACT_NAMES:-0}" == "1" ]]; then
    output_dir="$OUTPUT_ROOT/${method}_s${stage}"
    run_name="${method}_s${stage}"
  fi

  case "$method" in
    inpo)
      train_model \
        "$INPO_CONFIG" \
        "$policy_model" \
        "$pref_dir" \
        "$output_dir" \
        "$run_name" \
        1 \
        --loss_type=inpo \
        --eta="$INPO_ETA" \
        --ratio="$INPO_RATIO" \
        --beta="$INPO_BETA"
      ;;
    sppo)
      train_model \
        "$SPPO_CONFIG" \
        "$policy_model" \
        "$pref_dir" \
        "$output_dir" \
        "$run_name" \
        1 \
        --loss_type=sppo \
        --eta="$SPPO_ETA" \
        --ratio="$SPPO_RATIO" \
        --beta="$SPPO_BETA"
      ;;
    *)
      echo "Unknown avg baseline method '$method'. Supported: inpo,sppo" >&2
      exit 1
      ;;
  esac
}

for stage in $STAGES; do
  if [[ "$SHARE_STAGE1_BASE_DATA" == "1" && "$stage" -eq 1 ]]; then
    prepare_base_stage1_data
  fi

  if [[ "$RUN_HTMNPO" == "1" ]]; then
    for player in "${PLAYER_LIST[@]}"; do
      run_ht_player_stage "$player" "$stage"
    done
  fi

  if [[ "$RUN_RONPO" == "1" ]]; then
    run_ronpo_stage "$stage"
  fi

  if [[ "$RUN_INPO" == "1" ]]; then
    run_avg_baseline_stage inpo "$stage"
  fi

  if [[ "$RUN_SPPO" == "1" ]]; then
    run_avg_baseline_stage sppo "$stage"
  fi
done
