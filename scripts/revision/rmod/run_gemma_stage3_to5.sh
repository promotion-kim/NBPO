#!/usr/bin/env bash
# Continue the locked Gemma RONPO chain through Stage 5, fail-closed on each gate.
set -euo pipefail
SJ=/NHNHOME/AIPR/sjkim
P=$SJ/MNPO_rev_20260720
ROOT=$SJ/rmod_stage5_kappa_large_20260722
mkdir -p "$ROOT/logs"

passed() {
  "$SJ/venv_clean/bin/python" - "$1" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
raise SystemExit(0 if x.get("passed") is True and x.get("status") == "passed" else 1)
PY
}

for stage in 3 4 5; do
  work=$SJ/ronpo_gemma_s${stage}
  parent_stage=$((stage - 1))
  parent=$SJ/ronpo_gemma_s${parent_stage}/stage${parent_stage}
  if [[ ! -s "$work/precomputed/READY" ]]; then
    SOURCE=$parent WORK=$work GPUS=0,1,2,3 NPROC=4 \
      bash "$P/scripts/revision/rmod/prepare_gemma_stage_data.sh"
  fi
  if [[ ! -s "$work/smoke20/all_results.json" ]]; then
    GPU=0 GPUS=0,1,2,3 NPROC=4 MAX_STEPS=20 STAGE=$stage INIT=$parent \
      DATASET=$work/precomputed OUT=$work/smoke20 SEED=42 \
      bash "$P/scripts/revision/rmod/train_gemma_stage.sh"
  fi
  if [[ ! -s "$work/stage${stage}/all_results.json" ]]; then
    GPU=0 GPUS=0,1,2,3 NPROC=4 MAX_STEPS=1800 STAGE=$stage INIT=$parent \
      DATASET=$work/precomputed OUT=$work/stage${stage} SEED=42 \
      bash "$P/scripts/revision/rmod/train_gemma_stage.sh"
  fi
  if [[ ! -s "$work/eval/COMPLETE" ]]; then
    TAG=ronpo_gemma_s${stage} MODEL=$work/stage${stage} WORK=$work GPU=0 \
      bash "$P/scripts/revision/rmod/evaluate_gemma_policy.sh"
  fi
  if ! passed "$work/eval/stability_gate.json"; then
    printf 'stage=%s status=gate_failed time=%s\n' "$stage" "$(date -Is)" >> "$ROOT/fix_log.md"
    exit 1
  fi
  printf 'stage=%s status=complete time=%s\n' "$stage" "$(date -Is)" >> "$ROOT/logs/fig2_stage_chain_status.tsv"
done
date -Is > "$ROOT/FIGURE2_CHAIN_COMPLETE"
