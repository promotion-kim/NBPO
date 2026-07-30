#!/usr/bin/env bash
# Queue a fixed MMLU worker without touching an active process.
set -euo pipefail
[[ $# -ge 5 ]] || { echo "usage: $0 PROJECT STAGE4 OUTPUT GPU MODELS..." >&2; exit 2; }
PROJECT=$1; STAGE4=$2; OUTPUT=$3; GPU=$4; shift 4
while :; do
  idle=1
  for _ in 1 2 3; do
    pids=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
    [[ -z ${pids//[[:space:]]/} ]] || { idle=0; break; }
    sleep 3
  done
  [[ $idle -eq 1 ]] && break
  sleep 20
done
"$PROJECT/../venv_clean/bin/python" "$PROJECT/analysis/p10_saferlhf_training_seed43_20260718/run_p8_stage4_mmlu.py" \
  --project "$PROJECT" --stage4 "$STAGE4" --output "$OUTPUT" --gpu "$GPU" --models "$@"
