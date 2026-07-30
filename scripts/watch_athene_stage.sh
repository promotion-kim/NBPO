#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WATCH_INTERVAL="${WATCH_INTERVAL:-10}"
TARGET_DIR="${TARGET_DIR:-$PROJECT_ROOT/data/qwen2.5-1.5b-instruct_online_htmnpo_ronpo/htmnpo/athene/iter1/pairs}"
WATCH_LOG="${WATCH_LOG:-$PROJECT_ROOT/.athene_stage_watch.log}"

echo "Watching for Athene player stage..."
echo "Target: $TARGET_DIR"
echo "Interval: ${WATCH_INTERVAL}s"
echo "Log: $WATCH_LOG"

while true; do
  if [[ -d "$TARGET_DIR" ]]; then
    message="MNPO alert: Athene player stage has started. Press Ctrl+C in the training terminal now if you want to rescore Athene first."
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    printf '\a\a\a'
    printf '[%s] %s\n' "$timestamp" "$message" | tee -a "$WATCH_LOG"

    if command -v wall >/dev/null 2>&1; then
      printf '%s\n' "$message" | wall || true
    fi

    if command -v notify-send >/dev/null 2>&1; then
      notify-send "MNPO alert" "$message" || true
    fi

    exit 0
  fi
  sleep "$WATCH_INTERVAL"
done
