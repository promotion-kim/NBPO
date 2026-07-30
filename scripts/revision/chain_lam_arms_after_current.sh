#!/usr/bin/env bash
# Wait for the konly/aonly arms to finish on GPUs 1,2, then launch the
# lambda-sweep OS arms on the same GPUs so the whole sweep shares one stack.
set -uo pipefail
cd /home/sjkim/MNPO
LOGDIR=/ext_hdd/sjkim/mnpo/logs

while pgrep -f "run_mnpo.*os_ronpo_konly_k05" >/dev/null || pgrep -f "run_mnpo.*os_ronpo_aonly_k05" >/dev/null; do
  sleep 120
done

for arm in konly aonly; do
  if [[ ! -f "/ext_hdd/sjkim/mnpo/outputs_ronpo_h200/os_ronpo_${arm}_k05/all_results.json" ]]; then
    echo "[chain] ${arm} arm did not finish cleanly; not launching lambda arms" >&2
    exit 1
  fi
done

echo "[chain] konly/aonly done at $(date -Is); launching lambda arms"
GPU=1 TARGET_COL=target_os_k0p05_lam4 OUT=/ext_hdd/sjkim/mnpo/outputs_ronpo_h200/os_ronpo_os_k05_lam4 RUN=os-ronpo-os-k05-lam4 \
  DATASET=/ext_hdd/sjkim/mnpo/data/os_ronpo_iter2_targets_lam \
  nohup bash scripts/run_os_ronpo_arm.sh > "$LOGDIR/nohup_lam4_arm.log" 2>&1 &
GPU=2 TARGET_COL=target_os_k0p05_lam16 OUT=/ext_hdd/sjkim/mnpo/outputs_ronpo_h200/os_ronpo_os_k05_lam16 RUN=os-ronpo-os-k05-lam16 \
  DATASET=/ext_hdd/sjkim/mnpo/data/os_ronpo_iter2_targets_lam \
  nohup bash scripts/run_os_ronpo_arm.sh > "$LOGDIR/nohup_lam16_arm.log" 2>&1 &
wait
