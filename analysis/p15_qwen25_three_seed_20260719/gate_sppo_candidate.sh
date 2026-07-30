#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 7 ]] || { echo "usage: $0 PROJECT PY ROOT SOURCE CACHE CANDIDATE GPU" >&2; exit 2; }
PROJECT=$1 PY=$2 ROOT=$3 SOURCE=$4 CACHE=$5 CAND=$6 GPU=$7
S=$PROJECT/analysis/p14_qwen25_7b_stage4_seeds_20260718
bash "$S/decode_gate_repair.sh" "$PROJECT" "$PY" "$PY" "$ROOT/sppo_extension" "$SOURCE" "$CACHE" "$CAND" "$GPU" 3

