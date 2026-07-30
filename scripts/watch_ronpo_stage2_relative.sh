#!/usr/bin/env bash
set -euo pipefail

cd /home/sjkim/MNPO

TARGET_SESSION="${TARGET_SESSION:-ronpo_s2_rel}"
GEN_ROOT="${GEN_ROOT:-/ext_hdd/sjkim/mnpo/eval_generations/ronpo_stage2_relative_v1}"
STATE_DIR="${STATE_DIR:-/ext_hdd/sjkim/mnpo/watch_state/ronpo_s2_rel}"
LOG_DIR="${LOG_DIR:-/ext_hdd/sjkim/mnpo/logs}"
CHECK_INTERVAL="${CHECK_INTERVAL:-120}"
STORAGE_PATH="${STORAGE_PATH:-/ext_hdd}"
PYTHON_CHECK="${PYTHON_CHECK:-/home/sjkim/anaconda3/envs/mnpo_train/bin/python}"
LOG_FILE="$LOG_DIR/ronpo_s2_rel_watch_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$STATE_DIR" "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_FILE"
}

target_pane_id() {
  tmux list-panes -a -F '#{session_name} #{pane_id}' \
    | awk -v session="$TARGET_SESSION" '$1 == session { print $2; exit }'
}

log "watch start: target_session=$TARGET_SESSION gen_root=$GEN_ROOT"

while true; do
  if ! tmux has-session -t "=$TARGET_SESSION" 2>/dev/null; then
    log "target session ended; watcher exiting"
    exit 0
  fi

  df -h "$STORAGE_PATH" | tee -a "$LOG_FILE"

  while IFS= read -r gen_file; do
    step_dir="$(basename "$(dirname "$gen_file")")"
    marker="$STATE_DIR/${step_dir}.checked"
    if [[ -f "$marker" ]]; then
      continue
    fi

    log "checking collapse for $gen_file"
    result="$("$PYTHON_CHECK" scripts/check_generation_collapse.py "$gen_file" 2>&1 | tee -a "$LOG_FILE")"
    printf '%s\n' "$result" > "$marker"

    if grep -q 'status=collapse_suspect' "$marker"; then
      log "collapse suspected at $step_dir; stopping only target tmux session $TARGET_SESSION"
      pane_id="$(target_pane_id)"
      if [[ -n "$pane_id" ]]; then
        tmux send-keys -t "$pane_id" C-c
      else
        log "could not resolve exact target pane for $TARGET_SESSION"
      fi
      exit 2
    fi
  done < <(find "$GEN_ROOT" -maxdepth 4 -type f -name policy_generations.jsonl | sort)

  sleep "$CHECK_INTERVAL"
done
