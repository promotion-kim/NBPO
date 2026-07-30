#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 3 ]] || { echo "usage: $0 PYTHON PROJECT ROOT" >&2; exit 2; }
PY=$1; PROJECT=$2; ROOT=$3
specs=(43:ronpo_os 44:ronpo_os 43:dpo 43:inpo_avg 44:inpo_avg 44:dpo 43:sppo_avg 44:sppo_avg 43:ipo 43:simpo 44:simpo 44:ipo)
for spec in "${specs[@]}"; do
  seed=${spec%%:*}; arm=${spec#*:}; gate=$ROOT/seeds/s$seed/stage1/gates/$arm.json
  while [[ ! -s $gate ]]; do sleep 15; done
done
for spec in "${specs[@]}"; do
  seed=${spec%%:*}; arm=${spec#*:}
  "$PY" "$PROJECT/analysis/p14_qwen25_7b_stage4_seeds_20260718/finalize_gate_prereq_repair.py" \
    --root "$ROOT" --seed "$seed" --arm "$arm"
done
