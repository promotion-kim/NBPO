#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-results/ENV.txt}"
PYTHON_BIN="${PYTHON_BIN:-/home/sjkim/anaconda3/envs/mnpo_train/bin/python}"

mkdir -p "$(dirname "$OUT")"
{
  echo "# RONPO AAAI revision environment"
  echo "timestamp=$(date -Is)"
  echo "hostname=$(hostname)"
  echo "python=$($PYTHON_BIN -c 'import sys; print(sys.executable)')"
  echo
  echo "## pip freeze"
  "$PYTHON_BIN" -m pip freeze
  echo
  echo "## repo"
  git rev-parse HEAD 2>/dev/null || true
  git status --short 2>/dev/null || true
} > "$OUT"
echo "$OUT"
