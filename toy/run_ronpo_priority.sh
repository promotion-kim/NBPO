#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTDIR="${OUTDIR:-$PROJECT_ROOT/toy/toy_outputs_v2}"
EXPERIMENT="${EXPERIMENT:-all_priority}"
NUM_SEEDS="${NUM_SEEDS:-10}"

"$PYTHON_BIN" "$PROJECT_ROOT/toy/toy_v2.py" \
  --experiment "$EXPERIMENT" \
  --num_seeds "$NUM_SEEDS" \
  --outdir "$OUTDIR"
