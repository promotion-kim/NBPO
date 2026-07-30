#!/usr/bin/env bash
# Phase 1: wait for konly/aonly arms -> controlled eval of {base,topmass,os,konly,aonly}
#          -> launch lambda arms on the freed GPUs.
# Phase 2: wait for lambda arms -> controlled eval of {base,os,lam4,lam16}.
set -uo pipefail
cd /home/sjkim/MNPO
LOGDIR=/ext_hdd/sjkim/mnpo/logs
OUTROOT=/ext_hdd/sjkim/mnpo/outputs_ronpo_h200

while pgrep -f "run_mnpo.*os_ronpo_konly_k05" >/dev/null || pgrep -f "run_mnpo.*os_ronpo_aonly_k05" >/dev/null; do
  sleep 120
done
for arm in konly aonly; do
  if [[ ! -f "$OUTROOT/os_ronpo_${arm}_k05/all_results.json" ]]; then
    echo "[chain] ${arm} arm missing all_results.json; aborting" >&2
    exit 1
  fi
done

echo "[chain] arms done at $(date -Is); running factorized eval"
NEW_MODELS="konly=$OUTROOT/os_ronpo_konly_k05 aonly=$OUTROOT/os_ronpo_aonly_k05" \
WORK_DIR=/ext_hdd/sjkim/mnpo/eval/ronpo_factorized_stage2_20260720 \
GPUS="1 2" \
bash scripts/revision/eval_factorized_arms.sh > "$LOGDIR/eval_factorized_20260720.log" 2>&1
echo "[chain] factorized eval exit=$? at $(date -Is)"

echo "[chain] launching lambda arms"
GPU=1 TARGET_COL=target_os_k0p05_lam4 OUT=$OUTROOT/os_ronpo_os_k05_lam4 RUN=os-ronpo-os-k05-lam4 \
  DATASET=/ext_hdd/sjkim/mnpo/data/os_ronpo_iter2_targets_lam \
  nohup bash scripts/run_os_ronpo_arm.sh > "$LOGDIR/nohup_lam4_arm.log" 2>&1 &
GPU=2 TARGET_COL=target_os_k0p05_lam16 OUT=$OUTROOT/os_ronpo_os_k05_lam16 RUN=os-ronpo-os-k05-lam16 \
  DATASET=/ext_hdd/sjkim/mnpo/data/os_ronpo_iter2_targets_lam \
  nohup bash scripts/run_os_ronpo_arm.sh > "$LOGDIR/nohup_lam16_arm.log" 2>&1 &
sleep 300

while pgrep -f "run_mnpo.*os_ronpo_os_k05_lam4" >/dev/null || pgrep -f "run_mnpo.*os_ronpo_os_k05_lam16" >/dev/null; do
  sleep 120
done
for arm in lam4 lam16; do
  if [[ ! -f "$OUTROOT/os_ronpo_os_k05_${arm}/all_results.json" ]]; then
    echo "[chain] ${arm} arm missing all_results.json; aborting" >&2
    exit 1
  fi
done

echo "[chain] lambda arms done at $(date -Is); running lambda eval"
NEW_MODELS="lam4=$OUTROOT/os_ronpo_os_k05_lam4 lam16=$OUTROOT/os_ronpo_os_k05_lam16" \
REUSE="baseline os" \
WORK_DIR=/ext_hdd/sjkim/mnpo/eval/ronpo_lambda_stage2_20260721 \
GPUS="1 2" \
bash scripts/revision/eval_factorized_arms.sh > "$LOGDIR/eval_lambda_20260721.log" 2>&1
echo "[chain] lambda eval exit=$? at $(date -Is); all done"
