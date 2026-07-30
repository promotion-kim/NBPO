#!/usr/bin/env bash
set -euo pipefail

SRC="${SRC:?set SRC to the isolated local run root}"
DST="${DST:?set DST to the destination directory}"
LOG_GLOB="${LOG_GLOB:-$SRC/logs/*.log}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"

case "$SRC" in
  /home/sjkim/mnpo_loki3/*) ;;
  *)
    echo "Refusing to prune unexpected SRC=$SRC" >&2
    exit 2
    ;;
esac

case "$DST" in
  /ext_hdd/sjkim/mnpo/*) ;;
  *)
    echo "Refusing to copy to unexpected DST=$DST" >&2
    exit 2
    ;;
esac

echo "[watch] src=$SRC"
echo "[watch] dst=$DST"
echo "[watch] logs=$LOG_GLOB"

while true; do
  if compgen -G "$LOG_GLOB" >/dev/null; then
    if grep -E "Traceback|OutOfMemory|CUDA out of memory|ERROR|\\[rank[0-9]+\\]:.*Error" $LOG_GLOB >/dev/null 2>&1; then
      echo "[watch] failure marker detected; leaving SRC in place for debugging"
      exit 1
    fi
    if grep -F "[done]" $LOG_GLOB >/dev/null 2>&1; then
      echo "[watch] completion detected at $(date -Is)"
      mkdir -p "$DST"
      rsync -a "$SRC"/ "$DST"/
      echo "[watch] copy complete; pruning isolated source root"
      rm -rf "$SRC"
      echo "[watch] done at $(date -Is)"
      exit 0
    fi
  fi
  sleep "$INTERVAL_SECONDS"
done
