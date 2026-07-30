#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/MNPO_rev_20260710}"
RUN_ROOT="${RUN_ROOT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/baseline_repair_1p5b_20260714}"
LOG="$RUN_ROOT/logs/dispatch_grid.log"
cd "$PROJECT_ROOT"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] waiting for the two candidate-a training sessions"
while tmux has-session -t repair_sppo_a 2>/dev/null || tmux has-session -t repair_inpo_a 2>/dev/null; do
  sleep 30
done

for method in sppo inpo; do
  current="$RUN_ROOT/candidates/repair1p5b_${method}_a_s42"
  if [[ ! -f "$current/model.safetensors" || ! -f "$current/train_results.json" ]]; then
    echo "[$(date -Is)] ${method}-a did not finish cleanly; stopping downstream grid" >&2
    exit 3
  fi
done

echo "[$(date -Is)] launching the remaining symmetric four-candidate wave"
bash scripts/revision/run_baseline_repair_1p5b_candidate.sh \
  --method sppo --candidate b --gpu 0 \
  --eta 0.01 --reference-anchor 0.02 --sft-anchor 0.002 --learning-rate 5e-7 & p0=$!
bash scripts/revision/run_baseline_repair_1p5b_candidate.sh \
  --method sppo --candidate d --gpu 1 \
  --eta 0.01 --reference-anchor 0.05 --sft-anchor 0.005 --learning-rate 2.5e-7 & p1=$!
bash scripts/revision/run_baseline_repair_1p5b_candidate.sh \
  --method inpo --candidate b --gpu 2 \
  --eta 0.01 --reference-anchor 0.02 --sft-anchor 0.002 --learning-rate 5e-7 & p2=$!
bash scripts/revision/run_baseline_repair_1p5b_candidate.sh \
  --method inpo --candidate d --gpu 3 \
  --eta 0.01 --reference-anchor 0.05 --sft-anchor 0.005 --learning-rate 2.5e-7 & p3=$!

failed=0
for pid in "$p0" "$p1" "$p2" "$p3"; do
  wait "$pid" || failed=1
done
if [[ "$failed" -ne 0 ]]; then
  echo "[$(date -Is)] at least one remaining candidate failed; preserving all artifacts" >&2
  exit 4
fi

echo "[$(date -Is)] launching the final symmetric candidate-c pair"
bash scripts/revision/run_baseline_repair_1p5b_candidate.sh \
  --method sppo --candidate c --gpu 0,1 \
  --eta 0.005 --reference-anchor 0.05 --sft-anchor 0.005 --learning-rate 2.5e-7 & p4=$!
bash scripts/revision/run_baseline_repair_1p5b_candidate.sh \
  --method inpo --candidate c --gpu 2,3 \
  --eta 0.005 --reference-anchor 0.05 --sft-anchor 0.005 --learning-rate 2.5e-7 & p5=$!
failed=0
for pid in "$p4" "$p5"; do
  wait "$pid" || failed=1
done
if [[ "$failed" -ne 0 ]]; then
  echo "[$(date -Is)] at least one candidate-c run failed; preserving all artifacts" >&2
  exit 5
fi
echo "[$(date -Is)] completed candidates a,b,c,d for SPPO and INPO"
