#!/usr/bin/env bash
# Start the locked joint-panel curve evaluation after the current seed-43 Stage-4 work releases the GPUs.
set -euo pipefail
[[ $# -eq 6 ]] || { echo "usage: $0 PROJECT OUTPUT P10_STAGE4 HOST GPU_LIST ROLE" >&2; exit 2; }
PROJECT=$1; OUT=$2; P10_STAGE4=$3; HOST=$4; GPU_LIST=$5; ROLE=$6
VENV=$PROJECT/../venv_clean
SCRIPT=$PROJECT/analysis/p11_saferlhf_stage_curve_20260718
ARMS=(ronpo_os ronpo_topmass inpo_avg sppo_avg simpo ipo dpo ht_mnpo_harmless ht_mnpo_helpfulness)
if [[ "$HOST" == h1 ]]; then
  WAIT_ARMS=(ronpo_os ronpo_topmass inpo_avg simpo ipo ht_mnpo_harmless ht_mnpo_helpfulness)
else
  WAIT_ARMS=(dpo sppo_avg)
fi
mkdir -p "$OUT/logs/launch" "$OUT/done"
while :; do
  all_stage4_gates=1
  for arm in "${WAIT_ARMS[@]}"; do
    [[ -s "$P10_STAGE4/stage4_stability_p8_locked_panel/gates/$arm.json" ]] || { all_stage4_gates=0; break; }
  done
  [[ $all_stage4_gates -eq 1 ]] && break

  # A genuine fail-closed Stage-3 outcome prevents the Stage-4 lock by
  # construction. Treat that terminal state as a resource barrier rather
  # than waiting forever for Stage-4 gate files that cannot be produced.
  stage3_gate_root=${P10_STAGE4/stage4_seed43/stage3_seed43}/stage3_stability_p8_locked_panel/gates
  all_stage3_gates=1; any_stage3_failure=0
  for arm in "${ARMS[@]}"; do
    gate=$stage3_gate_root/$arm.json
    [[ -s "$gate" ]] || { all_stage3_gates=0; break; }
    "$VENV/bin/python" - "$gate" <<'PY' || any_stage3_failure=1
import json, sys
d = json.load(open(sys.argv[1]))
raise SystemExit(0 if d.get("passed") is True else 1)
PY
  done
  [[ $all_stage3_gates -eq 1 && $any_stage3_failure -eq 1 ]] && break

  # Likewise, a failed Stage-4 training run writes a terminal job_status but
  # cannot produce a gate. Once every arm has a terminal status, the idle-GPU
  # check below safely hands the host to the stage-curve evaluation.
  all_stage4_statuses=1
  for arm in "${WAIT_ARMS[@]}"; do
    [[ -s "$P10_STAGE4/stage4/$arm/train/full/job_status.json" ]] || { all_stage4_statuses=0; break; }
  done
  [[ $all_stage4_statuses -eq 1 ]] && break
  sleep 30
done
IFS=, read -r -a GPUS <<< "$GPU_LIST"
while :; do
  idle=1
  for gpu in "${GPUS[@]}"; do
    for _ in 1 2 3; do
      pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
      [[ -z ${pids//[[:space:]]/} ]] || { idle=0; break 2; }
      sleep 3
    done
  done
  [[ $idle -eq 1 ]] && break
  sleep 30
done
if [[ "$HOST" == h1 ]]; then
  # Reused generations are gated outcome-blind before the new decode workers.
  "$SCRIPT/decode_and_gate.sh" "$PROJECT" "$OUT" base "$(awk -F '\t' '$1=="base" {print $4}' "$OUT/models.tsv")" "${GPUS[0]}"
  for method in ronpo_os inpo_avg sppo_avg simpo ipo dpo ht_mnpo_harmless ht_mnpo_helpfulness; do
    key=${method}_stage4
    "$SCRIPT/decode_and_gate.sh" "$PROJECT" "$OUT" "$key" "$(awk -F '\t' -v key="$key" '$1==key {print $4}' "$OUT/models.tsv")" "${GPUS[0]}"
  done
  sets=(
    "ronpo_os_stage1 inpo_avg_stage1 ronpo_os_stage2 inpo_avg_stage2"
    "sppo_avg_stage1 simpo_stage1 sppo_avg_stage2 simpo_stage2"
    "ipo_stage1 dpo_stage1 ipo_stage2 dpo_stage2"
    "ht_mnpo_harmless_stage1 ht_mnpo_helpfulness_stage1 ht_mnpo_harmless_stage2 ht_mnpo_helpfulness_stage2"
  )
else
  sets=(
    "ronpo_os_stage3 inpo_avg_stage3 sppo_avg_stage3 simpo_stage3"
    "ipo_stage3 dpo_stage3 ht_mnpo_harmless_stage3 ht_mnpo_helpfulness_stage3"
  )
fi
pids=()
for index in "${!sets[@]}"; do
  gpu=${GPUS[$index]}
  # shellcheck disable=SC2086
  "$SCRIPT/decode_worker.sh" "$PROJECT" "$OUT" "$gpu" "$OUT/done/decode_${HOST}_${gpu}" ${sets[$index]} > "$OUT/logs/launch/decode_${HOST}_${gpu}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
date -Is > "$OUT/done/DECODE_DONE_${HOST}"
if [[ "$HOST" == h1 ]]; then
  while [[ ! -f "$OUT/done/DECODE_DONE_h2" ]]; do sleep 20; done
  "$VENV/bin/python" "$SCRIPT/prepare_joint_pool.py" --project "$PROJECT" --output "$OUT" > "$OUT/logs/launch/prepare_joint_pool.log" 2>&1
  score_pids=()
  for shard in 0 1 2 3; do
    gpu=${GPUS[$shard]}
    "$SCRIPT/score_worker.sh" "$PROJECT" "$OUT" "$gpu" "$shard" > "$OUT/logs/launch/score_h1_${shard}.log" 2>&1 & score_pids+=("$!")
  done
  for pid in "${score_pids[@]}"; do wait "$pid"; done
  date -Is > "$OUT/done/SCORE_DONE_h1"
  while [[ ! -f "$OUT/done/SCORE_DONE_h2" ]]; do sleep 20; done
  "$VENV/bin/python" "$SCRIPT/finalize_stage_curve.py" --project "$PROJECT" --output "$OUT" > "$OUT/logs/launch/finalize.log" 2>&1
  date -Is > "$OUT/COMPLETED"
else
  while [[ ! -f "$OUT/SCORE_READY" ]]; do sleep 20; done
  score_pids=()
  for index in 0 1; do
    shard=$((index+4)); gpu=${GPUS[$index]}
    "$SCRIPT/score_worker.sh" "$PROJECT" "$OUT" "$gpu" "$shard" > "$OUT/logs/launch/score_h2_${shard}.log" 2>&1 & score_pids+=("$!")
  done
  for pid in "${score_pids[@]}"; do wait "$pid"; done
  date -Is > "$OUT/done/SCORE_DONE_h2"
fi
