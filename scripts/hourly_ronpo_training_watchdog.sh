#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-/ext_hdd/sjkim/mnpo/experiments/ronpo_safety_conflict_qwen25_1p5b_20260629}"
PYTHON_TRAIN="${PYTHON_TRAIN:-/home/sjkim/anaconda3/envs/mnpo_train/bin/python}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-3600}"
STOP_AT="${STOP_AT:-2026-07-02 08:00:00}"
CHECKED_DIR="${CHECKED_DIR:-/tmp/ronpo_hourly_watchdog}"
LOG_FILE="${LOG_FILE:-$OUT_ROOT/logs/hourly_watchdog_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$CHECKED_DIR" "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[watchdog] start $(date -Is)"
echo "[watchdog] stop_at=$STOP_AT interval=${INTERVAL_SECONDS}s"
echo "[watchdog] out_root=$OUT_ROOT"
echo "[watchdog] log=$LOG_FILE"

declare -a SPECS=(
  "ronpo-safe-expect-ckptfull-s42|ronpo_p1_s42_gpu0|0|42"
  "ronpo-safe-expect-ckptfull-s43|ronpo_p1_s43_gpu1|1|43"
  "ronpo-safe-expect-ckptfull-s44|ronpo_p1_s44_gpu2|2|44"
)

latest_log_for() {
  local run_name="$1" gpu="$2" seed="$3"
  find "$OUT_ROOT/logs" -maxdepth 1 -type f -name "${run_name}_gpu${gpu}_seed${seed}_*.log" -printf '%T@ %p\n' 2>/dev/null \
    | sort -n | tail -1 | cut -d' ' -f2-
}

latest_checkpoint_for() {
  local run_name="$1" seed="$2"
  local output_dir="$OUT_ROOT/outputs/${run_name}_seed${seed}"
  find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f %p\n' 2>/dev/null \
    | sort -t- -k2,2n | tail -1 | cut -d' ' -f2-
}

is_done() {
  local run_name="$1" seed="$2" gpu="$3"
  local output_dir="$OUT_ROOT/outputs/${run_name}_seed${seed}"
  local log_file
  log_file="$(latest_log_for "$run_name" "$gpu" "$seed")"
  [[ -f "$output_dir/train_results.json" ]] && return 0
  [[ -n "$log_file" ]] && grep -q '\*\*\* Training complete' "$log_file" && return 0
  [[ -n "$log_file" ]] && grep -q "^\[done\].*mode=full_expect" "$log_file" && return 0
  return 1
}

latest_collapse_status() {
  local run_name="$1"
  local gen_root="$OUT_ROOT/eval_generations/$run_name"
  [[ -d "$gen_root" ]] || {
    echo "status=missing_generation"
    return 0
  }

  local latest_jsonl
  latest_jsonl="$(find "$gen_root" -path '*/policy_generations.jsonl' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)"
  [[ -n "$latest_jsonl" ]] || {
    echo "status=missing_generation"
    return 0
  }
  "$PYTHON_TRAIN" "$PROJECT_ROOT/scripts/check_generation_collapse.py" "$latest_jsonl"
}

best_ok_checkpoint() {
  local run_name="$1" seed="$2"
  local gen_root="$OUT_ROOT/eval_generations/$run_name"
  local best_step=""
  local jsonl result step ckpt

  [[ -d "$gen_root" ]] || return 1
  while IFS= read -r -d '' jsonl; do
    result="$("$PYTHON_TRAIN" "$PROJECT_ROOT/scripts/check_generation_collapse.py" "$jsonl")"
    [[ "$result" == status=ok* ]] || continue
    step="$(basename "$(dirname "$jsonl")" | sed 's/^step-//')"
    step="$((10#$step))"
    ckpt="$OUT_ROOT/outputs/${run_name}_seed${seed}/checkpoint-${step}"
    [[ -d "$ckpt" ]] || continue
    best_step="$step"
  done < <(find "$gen_root" -path '*/policy_generations.jsonl' -type f -print0 | sort -z)

  [[ -n "$best_step" ]] || return 1
  echo "$OUT_ROOT/outputs/${run_name}_seed${seed}/checkpoint-${best_step}"
}

restart_run() {
  local run_name="$1" session="$2" gpu="$3" seed="$4" reason="$5" resume_ckpt="${6:-}"

  echo "[watchdog] restart run=$run_name session=$session gpu=$gpu seed=$seed reason=$reason resume=${resume_ckpt:-none}"
  tmux send-keys -t "$session" C-c || true
  sleep 5
  tmux send-keys -t "$session" "cd $PROJECT_ROOT" C-m
  local cmd="RUN_NAME_OVERRIDE=$run_name TRAIN_BATCH_SIZE=4 TRAIN_EVAL_BATCH_SIZE=4 PRECOMPUTE_BATCH_SIZE=8 GRAD_ACCUM=4 SAVE_TOTAL_LIMIT=5 EVAL_STEPS=100 SAVE_STEPS=100 LOGGING_STEPS=5 EVAL_GENERATION_BACKEND=checkpoint EVAL_GENERATION_SAMPLES=5 EVAL_GENERATION_MAX_NEW_TOKENS=256 EVAL_GENERATION_DO_SAMPLE=false WANDB_RUN_GROUP=ronpo-phase1-selection-full"
  if [[ -n "$resume_ckpt" ]]; then
    cmd="$cmd RESUME_FROM_CHECKPOINT=$resume_ckpt"
  fi
  cmd="$cmd scripts/run_ronpo_safety_conflict_train.sh full_expect $gpu $seed"
  tmux send-keys -t "$session" "$cmd" C-m
}

while true; do
  now_epoch="$(date +%s)"
  stop_epoch="$(date -d "$STOP_AT" +%s)"
  if [[ "$now_epoch" -ge "$stop_epoch" ]]; then
    echo "[watchdog] stop $(date -Is): reached stop_at"
    exit 0
  fi

  echo
  echo "[watchdog] tick $(date -Is)"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits || true

  for spec in "${SPECS[@]}"; do
    IFS='|' read -r run_name session gpu seed <<< "$spec"
    echo "[watchdog] inspect run=$run_name session=$session gpu=$gpu seed=$seed"

    if is_done "$run_name" "$seed" "$gpu"; then
      echo "[watchdog] done run=$run_name"
      continue
    fi

    collapse_status="$(latest_collapse_status "$run_name")"
    echo "[watchdog] collapse $collapse_status"
    if [[ "$collapse_status" == status=collapse_suspect* ]]; then
      if resume_ckpt="$(best_ok_checkpoint "$run_name" "$seed")"; then
        restart_run "$run_name" "$session" "$gpu" "$seed" "collapse_suspect" "$resume_ckpt"
      else
        restart_run "$run_name" "$session" "$gpu" "$seed" "collapse_suspect_no_ok_checkpoint" ""
      fi
      continue
    fi

    if pgrep -af "mnpo_scripts.run_mnpo .*--run_name=${run_name}( |$)" >/dev/null || pgrep -af "mnpo_scripts.precompute .*precomputed_full_expect_seed${seed}( |$)" >/dev/null; then
      echo "[watchdog] process_ok run=$run_name"
      continue
    fi

    if resume_ckpt="$(latest_checkpoint_for "$run_name" "$seed")"; then
      restart_run "$run_name" "$session" "$gpu" "$seed" "process_missing" "$resume_ckpt"
    else
      restart_run "$run_name" "$session" "$gpu" "$seed" "process_missing_no_checkpoint" ""
    fi
  done

  sleep "$INTERVAL_SECONDS"
done
