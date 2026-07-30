#!/usr/bin/env bash
set -euo pipefail

cd /home/sjkim/MNPO

TARGET_SESSION="${TARGET_SESSION:-ronpo_s2_rel}"
WATCH_SESSION="${WATCH_SESSION:-ronpo_s2_rel_watch}"
DEADLINE="${DEADLINE:-2026-06-24 08:00:00}"
CHECK_INTERVAL="${CHECK_INTERVAL:-300}"
STORAGE_PATH="${STORAGE_PATH:-/ext_hdd}"
LOG_DIR="${LOG_DIR:-/ext_hdd/sjkim/mnpo/logs}"
SUP_STATE_DIR="${SUP_STATE_DIR:-/ext_hdd/sjkim/mnpo/watch_state/ronpo_s2_rel_supervisor}"
PYTHON_CHECK="${PYTHON_CHECK:-/home/sjkim/anaconda3/envs/mnpo_train/bin/python}"
RUN_SCRIPT="${RUN_SCRIPT:-/home/sjkim/MNPO/scripts/run_ronpo_stage2_relative_odin2.sh}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-/ext_hdd/sjkim/mnpo/outputs_ronpo_h200}"
BASE_GEN_ROOT="${BASE_GEN_ROOT:-/ext_hdd/sjkim/mnpo/eval_generations}"
CURRENT_TAG="${CURRENT_TAG:-lr1e7}"
CURRENT_LR="${CURRENT_LR:-1.0e-7}"
NEXT_TAG="${NEXT_TAG:-lr5e8}"
NEXT_LR="${NEXT_LR:-5.0e-8}"
RUN_HOST_SUFFIX="${RUN_HOST_SUFFIX:-od2}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-1}"
EVAL_GENERATION_CUDA_VISIBLE_DEVICES="${EVAL_GENERATION_CUDA_VISIBLE_DEVICES:-0}"
ACCELERATE_MAIN_PROCESS_PORT="${ACCELERATE_MAIN_PROCESS_PORT:-}"
TRAIN_PER_DEVICE_BATCH_SIZE="${TRAIN_PER_DEVICE_BATCH_SIZE:-4}"
TRAIN_PER_DEVICE_EVAL_BATCH_SIZE="${TRAIN_PER_DEVICE_EVAL_BATCH_SIZE:-4}"
TRAIN_GRADIENT_ACCUMULATION_STEPS="${TRAIN_GRADIENT_ACCUMULATION_STEPS:-4}"
TRAIN_SAVE_TOTAL_LIMIT="${TRAIN_SAVE_TOTAL_LIMIT:-5}"

mkdir -p "$LOG_DIR" "$SUP_STATE_DIR"
LOG_FILE="$LOG_DIR/ronpo_s2_rel_supervisor_$(date +%Y%m%d_%H%M%S).log"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_FILE"
}

target_pane_id() {
  tmux list-panes -a -F '#{session_name} #{pane_id}' \
    | awk -v session="$TARGET_SESSION" '$1 == session { print $2; exit }'
}

deadline_epoch() {
  date -d "$DEADLINE" +%s
}

now_epoch() {
  date +%s
}

gen_root_for_tag() {
  printf '%s/ronpo_stage2_relative_%s_%s\n' "$BASE_GEN_ROOT" "$1" "$RUN_HOST_SUFFIX"
}

output_dir_for_tag() {
  printf '%s/qwen2.5-1.5b-instruct_ronpo_stage2_relative_%s_%s\n' "$BASE_OUTPUT_ROOT" "$1" "$RUN_HOST_SUFFIX"
}

run_name_for_tag() {
  printf 'ronpo-s2-%s-%s\n' "$1" "$RUN_HOST_SUFFIX"
}

launch_training() {
  local tag="$1"
  local lr="$2"
  local gen_root output_dir run_name
  gen_root="$(gen_root_for_tag "$tag")"
  output_dir="$(output_dir_for_tag "$tag")"
  run_name="$(run_name_for_tag "$tag")"
  mkdir -p "$gen_root" "$output_dir"
  log "launching target session=$TARGET_SESSION tag=$tag lr=$lr"
  tmux new-session -d -s "$TARGET_SESSION" -c /home/sjkim/MNPO \
    "CUDA_VISIBLE_DEVICES=$TRAIN_CUDA_VISIBLE_DEVICES EVAL_GENERATION_CUDA_VISIBLE_DEVICES=$EVAL_GENERATION_CUDA_VISIBLE_DEVICES ACCELERATE_MAIN_PROCESS_PORT=$ACCELERATE_MAIN_PROCESS_PORT TRAIN_PER_DEVICE_BATCH_SIZE=$TRAIN_PER_DEVICE_BATCH_SIZE TRAIN_PER_DEVICE_EVAL_BATCH_SIZE=$TRAIN_PER_DEVICE_EVAL_BATCH_SIZE TRAIN_GRADIENT_ACCUMULATION_STEPS=$TRAIN_GRADIENT_ACCUMULATION_STEPS TRAIN_SAVE_TOTAL_LIMIT=$TRAIN_SAVE_TOTAL_LIMIT LEARNING_RATE=$lr OUTPUT_DIR=$output_dir GEN_DIR=$gen_root RUN_NAME=$run_name LOG_PREFIX=ronpo_s2_rel_${tag}_${RUN_HOST_SUFFIX} bash $RUN_SCRIPT"
}

launch_watcher() {
  local tag="$1"
  local gen_root state_dir
  gen_root="$(gen_root_for_tag "$tag")"
  state_dir="/ext_hdd/sjkim/mnpo/watch_state/ronpo_s2_rel_${tag}_${RUN_HOST_SUFFIX}"
  if tmux has-session -t "=$WATCH_SESSION" 2>/dev/null; then
    return 0
  fi
  log "launching watcher session=$WATCH_SESSION gen_root=$gen_root"
  tmux new-session -d -s "$WATCH_SESSION" -c /home/sjkim/MNPO \
    "TARGET_SESSION=$TARGET_SESSION GEN_ROOT=$gen_root STATE_DIR=$state_dir CHECK_INTERVAL=120 bash scripts/watch_ronpo_stage2_relative.sh"
}

check_collapse_files() {
  local tag="$1"
  local gen_root gen_file step_dir marker result
  gen_root="$(gen_root_for_tag "$tag")"
  [[ -d "$gen_root" ]] || return 0
  while IFS= read -r gen_file; do
    step_dir="$(basename "$(dirname "$gen_file")")"
    marker="$SUP_STATE_DIR/${tag}_${step_dir}.checked"
    [[ -f "$marker" ]] && continue
    log "supervisor checking collapse for $gen_file"
    result="$("$PYTHON_CHECK" scripts/check_generation_collapse.py "$gen_file" 2>&1 | tee -a "$LOG_FILE")"
    printf '%s\n' "$result" > "$marker"
    if grep -q 'status=collapse_suspect' "$marker"; then
      log "collapse suspected at tag=$tag $step_dir"
      return 2
    fi
  done < <(find "$gen_root" -maxdepth 4 -type f -name policy_generations.jsonl | sort)
  return 0
}

fatal_log_seen() {
  tmux capture-pane -t "$TARGET_SESSION" -p -S -240 2>/dev/null \
    | grep -E 'Traceback|CUDA out of memory|OutOfMemory|RuntimeError|UnicodeEncodeError|Killed|ChildFailedError' >/dev/null
}

log_status() {
  log "status check tag=$CURRENT_TAG"
  df -h "$STORAGE_PATH" | tee -a "$LOG_FILE"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits | tee -a "$LOG_FILE" || true
  if tmux has-session -t "=$TARGET_SESSION" 2>/dev/null; then
    tmux capture-pane -t "$TARGET_SESSION" -p -S -40 | tail -40 | tee -a "$LOG_FILE" >/dev/null || true
  else
    log "target session missing"
  fi
}

log "supervisor start target=$TARGET_SESSION watcher=$WATCH_SESSION deadline=$DEADLINE"

while (( "$(now_epoch)" < "$(deadline_epoch)" )); do
  log_status
  launch_watcher "$CURRENT_TAG"

  if tmux has-session -t "=$TARGET_SESSION" 2>/dev/null; then
    if fatal_log_seen; then
      log "fatal training log pattern detected; stopping target session for clean restart"
      pane_id="$(target_pane_id)"
      if [[ -n "$pane_id" ]]; then
        tmux send-keys -t "$pane_id" C-c || true
      fi
      sleep 20
      if tmux has-session -t "=$TARGET_SESSION" 2>/dev/null; then
        tmux kill-session -t "=$TARGET_SESSION" || true
      fi
      launch_training "$CURRENT_TAG" "$CURRENT_LR"
      launch_watcher "$CURRENT_TAG"
    fi
  else
    log "target session is not running; relaunching current tag=$CURRENT_TAG"
    launch_training "$CURRENT_TAG" "$CURRENT_LR"
    launch_watcher "$CURRENT_TAG"
  fi

  if ! check_collapse_files "$CURRENT_TAG"; then
    log "stopping collapsed target session"
    pane_id="$(target_pane_id)"
    if [[ -n "$pane_id" ]]; then
      tmux send-keys -t "$pane_id" C-c || true
    fi
    sleep 20
    if tmux has-session -t "=$TARGET_SESSION" 2>/dev/null; then
      tmux kill-session -t "=$TARGET_SESSION" || true
    fi
    if [[ "$CURRENT_TAG" != "$NEXT_TAG" ]]; then
      CURRENT_TAG="$NEXT_TAG"
      CURRENT_LR="$NEXT_LR"
      log "switching to conservative restart tag=$CURRENT_TAG lr=$CURRENT_LR"
      launch_training "$CURRENT_TAG" "$CURRENT_LR"
      if tmux has-session -t "=$WATCH_SESSION" 2>/dev/null; then
        tmux kill-session -t "=$WATCH_SESSION" || true
      fi
      launch_watcher "$CURRENT_TAG"
    else
      log "collapse repeated on conservative tag; leaving target stopped for manual analysis"
      exit 2
    fi
  fi

  sleep "$CHECK_INTERVAL"
done

log "supervisor reached deadline=$DEADLINE; exiting"
