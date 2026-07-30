#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 {sppo|inpo} GPU_PAIR" >&2
  exit 2
fi
METHOD="$1"
GPU_PAIR="$2"
[[ "$METHOD" =~ ^(sppo|inpo)$ ]] || exit 2
[[ "$GPU_PAIR" =~ ^[0-3],[0-3]$ ]] || exit 2

PROJECT_ROOT="${PROJECT_ROOT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/MNPO_rev_20260710}"
RUN_ROOT="${RUN_ROOT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/baseline_repair_1p5b_20260714}"
CURRENT_SESSION="repair_${METHOD}_a"
CURRENT="$RUN_ROOT/candidates/repair1p5b_${METHOD}_a_s42"
LOG="$RUN_ROOT/logs/dispatch_${METHOD}.log"
cd "$PROJECT_ROOT"

exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Is)] waiting for $CURRENT_SESSION"
while tmux has-session -t "$CURRENT_SESSION" 2>/dev/null; do
  sleep 30
done

if [[ ! -f "$CURRENT/model.safetensors" || ! -f "$CURRENT/train_results.json" ]]; then
  echo "[$(date -Is)] candidate a did not finish cleanly; not launching downstream $METHOD candidates" >&2
  exit 3
fi
"$RUN_ROOT/venv_train/bin/python" - "$CURRENT/train_results.json" <<'PY'
import json, math, sys
values = json.load(open(sys.argv[1]))
bad = {key: value for key, value in values.items() if isinstance(value, (int, float)) and not math.isfinite(value)}
if bad:
    raise SystemExit(f"non-finite train result: {bad}")
PY

echo "[$(date -Is)] launching ${METHOD}-b"
bash scripts/revision/run_baseline_repair_1p5b_candidate.sh \
  --method "$METHOD" --candidate b --gpu "$GPU_PAIR" \
  --eta 0.01 --reference-anchor 0.02 --sft-anchor 0.002 --learning-rate 5e-7

echo "[$(date -Is)] launching ${METHOD}-d"
bash scripts/revision/run_baseline_repair_1p5b_candidate.sh \
  --method "$METHOD" --candidate d --gpu "$GPU_PAIR" \
  --eta 0.01 --reference-anchor 0.05 --sft-anchor 0.005 --learning-rate 2.5e-7

echo "[$(date -Is)] completed ${METHOD} candidates a,b,d"
