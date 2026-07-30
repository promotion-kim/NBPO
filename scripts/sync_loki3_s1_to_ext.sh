#!/usr/bin/env bash
set -euo pipefail

SRC="${SRC:-/home/sjkim/mnpo_runs/loki3}"
DST="${DST:-/ext_hdd/sjkim/mnpo/loki3_s1}"
INTERVAL="${INTERVAL:-120}"

mkdir -p "$DST"

while true; do
  date -Is
  for name in logs out work wandb eval; do
    if [[ -d "$SRC/$name" ]]; then
      mkdir -p "$DST/$name"
      rsync -a --partial "$SRC/$name/" "$DST/$name/"
    fi
  done
  df -h "$DST" || true
  sleep "$INTERVAL"
done
