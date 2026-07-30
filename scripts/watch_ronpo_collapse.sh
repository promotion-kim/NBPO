#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $(( $# % 2 )) -ne 0 ]]; then
  echo "usage: $0 RUN_NAME TMUX_SESSION [RUN_NAME TMUX_SESSION ...]" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-/ext_hdd/sjkim/mnpo/experiments/ronpo_safety_conflict_qwen25_1p5b_20260629}"
PYTHON_TRAIN="${PYTHON_TRAIN:-/home/sjkim/anaconda3/envs/mnpo_train/bin/python}"
CHECKED_FILE="${CHECKED_FILE:-/tmp/ronpo_ckptfull_checked_generations.txt}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-120}"

touch "$CHECKED_FILE"

echo "[watch] start $(date -Is)"
echo "[watch] out_root=$OUT_ROOT"
echo "[watch] checked_file=$CHECKED_FILE"

while true; do
  idx=1
  while [[ $idx -le $# ]]; do
    run_name="${!idx}"
    idx=$((idx + 1))
    session="${!idx}"
    idx=$((idx + 1))

    gen_root="$OUT_ROOT/eval_generations/$run_name"
    [[ -d "$gen_root" ]] || continue

    while IFS= read -r -d '' jsonl; do
      if grep -Fxq "$jsonl" "$CHECKED_FILE"; then
        continue
      fi
      echo "[watch] checking $jsonl"
      result="$("$PYTHON_TRAIN" "$PROJECT_ROOT/scripts/check_generation_collapse.py" "$jsonl")"
      echo "[watch] $result"
      echo "$jsonl" >> "$CHECKED_FILE"
      if [[ "$result" == status=collapse_suspect* ]]; then
        echo "[watch] collapse suspected for $run_name; stopping tmux session $session"
        tmux send-keys -t "$session" C-c || true
      fi
    done < <(find "$gen_root" -path '*/policy_generations.jsonl' -type f -print0 | sort -z)
  done
  sleep "$INTERVAL_SECONDS"
done
