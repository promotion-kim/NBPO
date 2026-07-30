#!/usr/bin/env bash
# Use ronpo2 only when a completed checkpoint is ready for reward-blind gate/evaluation.
set -euo pipefail
SJ=/NHNHOME/AIPR/sjkim
P10=$SJ/MNPO_rev_20260710
P20=$SJ/MNPO_rev_20260720
F3=$P10/results/rmod_figure3_kappa_large_20260722

wait_completed() {
  local status=$1
  while [[ ! -s "$status" ]]; do sleep 1; done
  "$SJ/venv_clean/bin/python" - "$status" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
raise SystemExit(0 if x.get("status") == "completed" and x.get("returncode", 0) == 0 else 1)
PY
}

run_locked() {
  local gpu=$1
  shift
  flock -x "/tmp/ronpo2_figure_gpu${gpu}.lock" "$@"
}

figure3() {
  local stage label gpu exp arm status
  for stage in stage3 stage4; do
    pids=()
    for spec in k1:0 k2:1; do
      label=${spec%%:*}; gpu=${spec##*:}; arm=ronpo_os_${label}
      exp=$F3/$label/${stage}_run
      status=$exp/$stage/$arm/train/full/job_status.json
      wait_completed "$status"
      run_locked "$gpu" env TRAIN_SEED=42 RUN_PREFIX=fig3-${label} \
        bash "$P10/analysis/p10_saferlhf_training_seed43_20260718/decode_and_gate_continuation.sh" \
        "$P10" "$exp" "$stage" "$arm" "$gpu" & pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
  done
}

figure2() {
  local stage work status
  for stage in 4 5; do
    work=$SJ/ronpo_gemma_s${stage}
    status=$work/stage${stage}/all_results.json
    while [[ ! -s "$status" ]]; do sleep 30; done
    run_locked 0 env TAG=ronpo_gemma_s${stage} MODEL=$work/stage${stage} WORK=$work GPU=0 \
      bash "$P20/scripts/revision/rmod/evaluate_gemma_policy.sh"
  done
}

figure3 & p3=$!
figure2 & p2=$!
wait "$p3" "$p2"
date -Is > "$SJ/rmod_stage5_kappa_large_20260722/RONPO2_GATE_OFFLOAD_COMPLETE"
