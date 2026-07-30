#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT}"

PYTHON_TRAIN="${PYTHON_TRAIN:-/home/sjkim/anaconda3/envs/mnpo_train/bin/python}"
EXP_ROOT="${EXP_ROOT:-$PROJECT_ROOT/experiments/ronpo_aaai_hs2_qwen25_1p5b_20260706}"
SOURCE_SCORED="${SOURCE_SCORED:-$PROJECT_ROOT/experiments/ronpo_aaai_w0_qwen25_1p5b_armosafe_20260702/scored}"

mkdir -p "$EXP_ROOT/scored"
for split in train test; do
  for objective in helpfulness safety; do
    src="$SOURCE_SCORED/${split}_${objective}.jsonl"
    dst="$EXP_ROOT/scored/${split}_${objective}.jsonl"
    if [[ ! -f "$src" ]]; then
      echo "Missing scored file: $src" >&2
      exit 1
    fi
    if [[ ! -e "$dst" ]]; then
      ln -s "$src" "$dst"
    fi
  done
done

build_pairs() {
  local mode="$1"
  local selection="$2"
  local pairs_per_prompt="$3"
  local strategy="${4:-sigma}"
  local pair_mode="${5:-expected_relative_policy_vs_policy}"
  local pair_root="$EXP_ROOT/pairs/$mode"
  mkdir -p "$pair_root"

  for split in train test; do
    echo "[build] mode=$mode strategy=$strategy selection=$selection split=$split objectives=helpfulness,safety"
    "$PYTHON_TRAIN" "$PROJECT_ROOT/mnpo_scripts/build_multi_objective_dataset.py" \
      --scored_files \
        "helpfulness=$EXP_ROOT/scored/${split}_helpfulness.jsonl" \
        "safety=$EXP_ROOT/scored/${split}_safety.jsonl" \
      --mnpo_output "$pair_root/${split}_mnpo_average_unused.jsonl" \
      --ronpo_output "$pair_root/${split}_ronpo.jsonl" \
      --merged_output "$pair_root/${split}_merged_scores.jsonl" \
      --summary_output "$pair_root/${split}_summary.csv" \
      --normalization minmax \
      --ronpo_pair_strategy "$strategy" \
      --adversary_steps "${ADVERSARY_STEPS:-25}" \
      --adversary_alpha "${ADVERSARY_ALPHA:-1.0}" \
      --adversary_kappa "${ADVERSARY_KAPPA:-0.05}" \
      --preference_scale "${PREFERENCE_SCALE:-8.0}" \
      --policy_mode "${POLICY_MODE:-uniform}" \
      --pairs_per_prompt "$pairs_per_prompt" \
      --adversary_selection "$selection" \
      --ronpo_policy_pair_mode "$pair_mode" \
      --ronpo_policy_samples_per_atom "${RONPO_POLICY_SAMPLES_PER_ATOM:-1}" \
      --k_only_fixed_atom avg_worst \
      --k_only_response_mode "${K_ONLY_RESPONSE_MODE:-uniform}" \
      --common_pair_seed "${COMMON_PAIR_SEED:-42}"
  done
}

build_pairs full_expect all "${EXPECT_PAIRS_PER_PROMPT:-3}" sigma expected_relative_policy_vs_policy
build_pairs k_expect all "${EXPECT_PAIRS_PER_PROMPT:-3}" sigma_k_only expected_relative_policy_vs_policy
build_pairs uniform all "${EXPECT_PAIRS_PER_PROMPT:-3}" uniform expected_relative_policy_vs_policy
build_pairs maxmin_pointwise all "${EXPECT_PAIRS_PER_PROMPT:-3}" maxmin_pointwise expected_relative_policy_vs_policy

echo "[done] built H/S-only pairs under $EXP_ROOT/pairs"
