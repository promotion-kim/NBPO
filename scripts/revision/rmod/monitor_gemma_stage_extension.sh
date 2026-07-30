#!/usr/bin/env bash
set -uo pipefail

HERE=$(cd "$(dirname "$0")/../../.." && pwd)
BSSH=$HERE/nhn/bssh.sh
LOCAL=$HERE/results/rmod_stage_extension_20260721
SJ=/NHNHOME/AIPR/sjkim
P=$SJ/MNPO_rev_20260720
mkdir -p "$LOCAL/monitor"

remote_file() { "$BSSH" ronpo "test -f '$1'" >/dev/null 2>&1; }
session() { "$BSSH" "$1" "tmux has-session -t '$2'" >/dev/null 2>&1; }
running() { "$BSSH" "$1" "pgrep -af '$2' | grep -v 'pgrep -af'" >/dev/null 2>&1; }
launch() {
  local host=$1 name=$2 command=$3
  session "$host" "$name" && return 0
  "$BSSH" "$host" "tmux new-session -d -s '$name' \"cd $P && $command; rc=\\\$?; echo \\\$rc > $SJ/${name}.EXIT\"" >/dev/null
}
gate_passed() {
  "$BSSH" ronpo "grep -q '\"status\": \"passed\"' '$1'" >/dev/null 2>&1
}
gate_failed_under() {
  "$BSSH" ronpo "find '$1' -path '*/gates/*.json' -type f -exec grep -l '\"status\": \"failed\"' {} + 2>/dev/null | grep -q ." >/dev/null 2>&1
}

snapshot() {
  local ts out
  ts=$(date +%Y%m%dT%H%M%S%z)
  out=$LOCAL/monitor/$ts.txt
  {
    echo "timestamp=$(date -Is)"
    for host in ronpo ronpo2 polymer; do
      echo "== $host =="
      "$BSSH" "$host" 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; tmux ls 2>/dev/null || true; pgrep -af "run_mnpo|mnpo_scripts.precompute|on_policy_data_gen.decode|rm_armo_multihead" || true' 2>&1
    done
  } > "$out"
}

last_snapshot=0
while true; do
  now=$(date +%s)
  if (( now - last_snapshot >= 1800 )); then
    snapshot
    last_snapshot=$now
  fi

  # Restart the primary training from its last saved checkpoint if interrupted.
  if ! remote_file "$SJ/ronpo_gemma_s2/stage2/all_results.json"; then
    if ! running ronpo2 'output_dir=/NHNHOME/AIPR/sjkim/ronpo_gemma_s2/stage2'; then
      "$BSSH" ronpo2 'tmux kill-session -t gemma_s2_primary 2>/dev/null || true' >/dev/null
      launch ronpo2 gemma_s2_primary \
        "GPU=0 STAGE=2 INIT=$SJ/ronpo_gemma_20260720/kappa_arms/os_k0p05 DATASET=$SJ/ronpo_gemma_s2/precomputed OUT=$SJ/ronpo_gemma_s2/stage2 SEED=42 bash scripts/revision/rmod/train_gemma_stage.sh"
    fi
  fi

  # Resume the two locked Figure-3 kappa chains after infrastructure exits.
  # A genuine reward-blind gate failure remains terminal and is never retried.
  P10=$SJ/MNPO_rev_20260710
  KROOT=$P10/results/rmod_figure3_kappa_20260721
  for spec in \
    "k0p01 0.01 0 $P10/results/p4_8b_saferlhf_table4_20260717/train/w2_900steps/ronpo_os_entropy_0p15_diagnostic fig3_k001_chain" \
    "k0p5 0.5 1 $P10/results/p4_8b_saferlhf_table4_20260717/train/w2_900steps/ronpo_os_entropy_0p85_diagnostic fig3_k05_chain"; do
    set -- $spec; label=$1; kappa=$2; gpu=$3; parent=$4; name=$5
    if ! remote_file "$KROOT/$label/COMPLETE" && ! gate_failed_under "$KROOT/$label" && ! session ronpo "$name"; then
      "$BSSH" ronpo "rm -f '$KROOT/${label}_chain.exit'" >/dev/null
      launch ronpo "$name" \
        "cd $P10 && bash analysis/rmod_figure3_kappa_20260721/run_kappa_continuation.sh $P10 $KROOT $label $kappa $parent $gpu > $KROOT/${label}_chain.log 2>&1"
    fi
  done

  # Four baseline arms start as soon as the shared average-oracle precompute is ready.
  if remote_file "$SJ/ronpo_gemma_baselines_s1/precomputed/READY"; then
    "$BSSH" polymer 'tmux kill-session -t gemma_baseprep 2>/dev/null || true' >/dev/null
    arms=(inpo sppo mnpo dpo)
    for gpu in 0 1 2 3; do
      arm=${arms[$gpu]}
      if remote_file "$SJ/ronpo_gemma_baselines_s1/$arm/all_results.json"; then
        "$BSSH" polymer "tmux kill-session -t gemma_b_${arm} 2>/dev/null || true" >/dev/null
      fi
      if ! remote_file "$SJ/ronpo_gemma_baselines_s1/$arm/all_results.json" && \
         ! running polymer "output_dir=$SJ/ronpo_gemma_baselines_s1/$arm"; then
        "$BSSH" polymer "tmux kill-session -t gemma_b_${arm} 2>/dev/null || true" >/dev/null
        launch polymer "gemma_b_${arm}" \
          "ARM=$arm GPU=$gpu SEED=42 bash scripts/revision/rmod/train_gemma_baseline_arm.sh"
      fi
    done
  elif ! running polymer 'mnpo_scripts.precompute.*ronpo_gemma_baselines_s1'; then
    "$BSSH" polymer 'tmux kill-session -t gemma_baseprep 2>/dev/null || true' >/dev/null
    launch polymer gemma_baseprep \
      "GPUS=0,1,2,3 NPROC=4 bash scripts/revision/rmod/prepare_gemma_baselines_s1.sh"
  fi

  # The user fixed the training seed to 42. Keep the remaining Stage-1
  # baseline trainers alive, but do not launch seed replications or estimator
  # variants in the primary stage-extension workflow.
  for spec in 'simpo ronpo 2' 'ipo ronpo2 1'; do
    set -- $spec; arm=$1; host=$2; gpu=$3
    if remote_file "$SJ/ronpo_gemma_baselines_s1/precomputed/READY" && \
       ! remote_file "$SJ/ronpo_gemma_baselines_s1/$arm/all_results.json" && \
       ! running ronpo "output_dir=$SJ/ronpo_gemma_baselines_s1/$arm" && \
       ! running ronpo2 "output_dir=$SJ/ronpo_gemma_baselines_s1/$arm"; then
      launch "$host" "gemma_b_${arm}" \
        "ARM=$arm GPU=$gpu SEED=42 bash scripts/revision/rmod/train_gemma_baseline_arm.sh"
    fi
  done

  # Evaluate Stage 2 once and fail closed before Stage 3.
  if remote_file "$SJ/ronpo_gemma_s2/stage2/all_results.json"; then
    if ! remote_file "$SJ/ronpo_gemma_s2/eval/COMPLETE" && \
       ! remote_file "$SJ/ronpo_gemma_s2/eval/stability_gate.json"; then
      "$BSSH" ronpo2 'tmux kill-session -t gemma_s2_primary 2>/dev/null || true'
      launch ronpo2 gemma_s2_eval \
        "TAG=ronpo_gemma_s2 MODEL=$SJ/ronpo_gemma_s2/stage2 WORK=$SJ/ronpo_gemma_s2 GPU=0 bash scripts/revision/rmod/evaluate_gemma_policy.sh"
    fi
  fi

  # Once the two remaining baseline trainers finish, use GPU 1 to evaluate
  # baseline arms one at a time under the identical 647-prompt gate.
  if remote_file "$SJ/ronpo_gemma_baselines_s1/ipo/all_results.json" && \
     remote_file "$SJ/ronpo_gemma_baselines_s1/simpo/all_results.json" && \
     ! running ronpo2 'evaluate_gemma_policy.sh'; then
    for arm in inpo sppo mnpo dpo ipo simpo; do
      if remote_file "$SJ/ronpo_gemma_baselines_s1/$arm/all_results.json" && \
         ! remote_file "$SJ/ronpo_gemma_baselines_s1/$arm/eval/COMPLETE" && \
         ! remote_file "$SJ/ronpo_gemma_baselines_s1/$arm/eval/stability_gate.json"; then
        "$BSSH" ronpo2 "tmux kill-session -t gemma_eval_${arm} 2>/dev/null || true" >/dev/null
        launch ronpo2 "gemma_eval_${arm}" \
          "TAG=baseline_gemma_${arm}_s1 MODEL=$SJ/ronpo_gemma_baselines_s1/$arm WORK=$SJ/ronpo_gemma_baselines_s1/$arm GPU=1 bash scripts/revision/rmod/evaluate_gemma_policy.sh"
        break
      fi
    done
  fi

  # Stage 3 uses four polymer GPUs after the first baseline wave finishes.
  polymer_done=true
  for arm in inpo sppo mnpo dpo; do
    remote_file "$SJ/ronpo_gemma_baselines_s1/$arm/all_results.json" || polymer_done=false
  done
  if gate_passed "$SJ/ronpo_gemma_s2/eval/stability_gate.json" && $polymer_done; then
    if ! remote_file "$SJ/ronpo_gemma_s3/precomputed/READY"; then
      if ! running ronpo 'prepare_gemma_stage_data.*ronpo_gemma_s3' && \
         ! running polymer 'prepare_gemma_stage_data.*ronpo_gemma_s3'; then
        launch polymer gemma_s3_prep \
          "SOURCE=$SJ/ronpo_gemma_s2/stage2 WORK=$SJ/ronpo_gemma_s3 GPUS=0,1,2,3 NPROC=4 bash scripts/revision/rmod/prepare_gemma_stage_data.sh"
      fi
    elif ! remote_file "$SJ/ronpo_gemma_s3/smoke20/all_results.json"; then
      launch polymer gemma_s3_train \
        "GPU=0 GPUS=0,1,2,3 NPROC=4 MAX_STEPS=20 STAGE=3 INIT=$SJ/ronpo_gemma_s2/stage2 DATASET=$SJ/ronpo_gemma_s3/precomputed OUT=$SJ/ronpo_gemma_s3/smoke20 SEED=42 bash scripts/revision/rmod/train_gemma_stage.sh"
    elif ! remote_file "$SJ/ronpo_gemma_s3/stage3/all_results.json"; then
      launch polymer gemma_s3_train \
        "GPU=0 GPUS=0,1,2,3 NPROC=4 MAX_STEPS=1800 STAGE=3 INIT=$SJ/ronpo_gemma_s2/stage2 DATASET=$SJ/ronpo_gemma_s3/precomputed OUT=$SJ/ronpo_gemma_s3/stage3 SEED=42 bash scripts/revision/rmod/train_gemma_stage.sh"
    elif ! remote_file "$SJ/ronpo_gemma_s3/eval/COMPLETE" && \
         ! remote_file "$SJ/ronpo_gemma_s3/eval/stability_gate.json"; then
      "$BSSH" polymer 'tmux kill-session -t gemma_s3_train 2>/dev/null || true'
      launch polymer gemma_s3_eval \
        "TAG=ronpo_gemma_s3 MODEL=$SJ/ronpo_gemma_s3/stage3 WORK=$SJ/ronpo_gemma_s3 GPU=0 bash scripts/revision/rmod/evaluate_gemma_policy.sh"
    fi
  fi

  # Stage 4 uses ronpo after the seed-43 and baseline jobs on that host finish.
  ronpo_free=true
  for name in gemma_b_simpo; do
    session ronpo "$name" && ronpo_free=false
  done
  if gate_passed "$SJ/ronpo_gemma_s3/eval/stability_gate.json" && $ronpo_free; then
    if ! remote_file "$SJ/ronpo_gemma_s4/precomputed/READY"; then
      launch ronpo gemma_s4_prep \
        "SOURCE=$SJ/ronpo_gemma_s3/stage3 WORK=$SJ/ronpo_gemma_s4 GPUS=0,1,2,3 NPROC=4 bash scripts/revision/rmod/prepare_gemma_stage_data.sh"
    elif ! remote_file "$SJ/ronpo_gemma_s4/smoke20/all_results.json"; then
      launch ronpo gemma_s4_train \
        "GPU=0 GPUS=0,1,2,3 NPROC=4 MAX_STEPS=20 STAGE=4 INIT=$SJ/ronpo_gemma_s3/stage3 DATASET=$SJ/ronpo_gemma_s4/precomputed OUT=$SJ/ronpo_gemma_s4/smoke20 SEED=42 bash scripts/revision/rmod/train_gemma_stage.sh"
    elif ! remote_file "$SJ/ronpo_gemma_s4/stage4/all_results.json"; then
      launch ronpo gemma_s4_train \
        "GPU=0 GPUS=0,1,2,3 NPROC=4 MAX_STEPS=1800 STAGE=4 INIT=$SJ/ronpo_gemma_s3/stage3 DATASET=$SJ/ronpo_gemma_s4/precomputed OUT=$SJ/ronpo_gemma_s4/stage4 SEED=42 bash scripts/revision/rmod/train_gemma_stage.sh"
    elif ! remote_file "$SJ/ronpo_gemma_s4/eval/COMPLETE" && \
         ! remote_file "$SJ/ronpo_gemma_s4/eval/stability_gate.json"; then
      "$BSSH" ronpo 'tmux kill-session -t gemma_s4_train 2>/dev/null || true'
      launch ronpo gemma_s4_eval \
        "TAG=ronpo_gemma_s4 MODEL=$SJ/ronpo_gemma_s4/stage4 WORK=$SJ/ronpo_gemma_s4 GPU=0 bash scripts/revision/rmod/evaluate_gemma_policy.sh"
    fi
  fi

  sleep 60
done
