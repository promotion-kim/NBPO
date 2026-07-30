#!/usr/bin/env bash
set -euo pipefail
PROJECT=${PROJECT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/venv_clean}
EXP=$PROJECT/results/p1_8b_hh_selection_20260716
source "$VENV/bin/activate"

while [[ ! -f "$EXP/scores/SCORE_COMPLETE" ]]; do
  if ! tmux has-session -t hh-score 2>/dev/null; then
    printf '%s scorer dispatcher ended without SCORE_COMPLETE\n' "$(date -Is)" >&2
    exit 1
  fi
  sleep 30
done

python "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/validate_score_artifacts.py" --root "$EXP" \
  >"$EXP/score_validation.log" 2>&1
cp "$EXP/score_validation.json" "$EXP/score_validation_precompact.json"
python "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/run_selection_analysis.py" --root "$EXP" \
  >"$EXP/analysis.log" 2>&1
python "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/render_selection_report.py" --root "$EXP"
sha256sum "$EXP/analysis_results/summary.json" >"$EXP/analysis_results/summary_precompact.sha256"

python "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/compact_score_artifacts.py" --root "$EXP" \
  >"$EXP/score_compaction.log" 2>&1
python "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/validate_score_artifacts.py" --root "$EXP" \
  >"$EXP/score_validation_compact.log" 2>&1
python "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/run_selection_analysis.py" --root "$EXP" \
  >"$EXP/analysis_postcompact.log" 2>&1
python "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/render_selection_report.py" --root "$EXP"
sha256sum "$EXP/analysis_results/summary.json" >"$EXP/analysis_results/summary_postcompact.sha256"
cut -d' ' -f1 "$EXP/analysis_results/summary_precompact.sha256" >"$EXP/.pre_hash"
cut -d' ' -f1 "$EXP/analysis_results/summary_postcompact.sha256" >"$EXP/.post_hash"
cmp "$EXP/.pre_hash" "$EXP/.post_hash"
rm -f "$EXP/.pre_hash" "$EXP/.post_hash"

{
  printf '%s\n' "$EXP/generations/ (14 policy/probe generation directories)"
  printf '%s\n' "$EXP/response_pool.jsonl"
} >"$EXP/deleted_raw_artifacts.txt"
rm -rf "$EXP/generations"
rm -f "$EXP/response_pool.jsonl"
date -Is >"$EXP/COMPLETE"
