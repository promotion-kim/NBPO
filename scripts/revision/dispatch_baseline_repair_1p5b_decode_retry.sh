#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/MNPO_rev_20260710}"
RUN_ROOT="${RUN_ROOT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/baseline_repair_1p5b_20260714}"
LOG="$RUN_ROOT/logs/decode_retry_dispatch.log"
cd "$PROJECT_ROOT"
exec > >(tee -a "$LOG") 2>&1

run_wave() {
  local method="$1"
  local -a pids=()
  local candidate gpu
  for spec in a:0 b:1 c:2 d:3; do
    candidate="${spec%%:*}"
    gpu="${spec##*:}"
    echo "[$(date -Is)] retry decode ${method}-${candidate} on authorized GPU ${gpu}"
    bash scripts/revision/run_baseline_repair_1p5b_decode.sh \
      "repair1p5b_${method}_${candidate}_s42" "$gpu" &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "[$(date -Is)] ${method} retry decode wave failed; preserving logs and stopping" >&2
    return 1
  fi
}

run_wave sppo
run_wave inpo
echo "[$(date -Is)] all eight retry decodes completed; entering unchanged gate/RM pipeline"
bash scripts/revision/dispatch_baseline_repair_1p5b_eval.sh

