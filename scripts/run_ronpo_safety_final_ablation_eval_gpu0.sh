#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUT_ROOT="${OUT_ROOT:-/ext_hdd/sjkim/mnpo/experiments/ronpo_safety_conflict_qwen25_1p5b_20260629}"
EXP_ROOT="${EXP_ROOT:-$PROJECT_ROOT/experiments/ronpo_safety_conflict_qwen25_1p5b_20260629}"

FULL_FINAL="${FULL_FINAL:-$OUT_ROOT/outputs/ronpo-safe-full-s1_seed42/checkpoint-3152}"
KONLY_FINAL="${KONLY_FINAL:-$OUT_ROOT/outputs/ronpo-safe-konly-s1_seed42/checkpoint-2227}"

export GPU="${GPU:-0}"
export WORK_DIR="${WORK_DIR:-$EXP_ROOT/reward_eval_final_ablation_$(date +%Y%m%d_%H%M%S)}"
export EVAL_SPECS="base=Qwen/Qwen2.5-1.5B-Instruct
full_final=$FULL_FINAL
konly_final=$KONLY_FINAL"

exec bash "$PROJECT_ROOT/scripts/run_ronpo_safety_reward_eval_gpu0.sh"
