#!/usr/bin/env bash
# Wait for the local konly/aonly arms, then run the controlled 5-policy eval.
# The lambda arms moved to the B200 cluster, so this chain ends after the eval
# and frees GPUs 1,2 for the IFEval suite of the new arms.
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
echo "[chain] factorized eval exit=$? at $(date -Is); GPUs 1,2 free for IFEval"
