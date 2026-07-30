#!/usr/bin/env bash
# Wait for the local konly arm and the transferred aonly-b200 checkpoint, then
# run the controlled 5-policy eval on GPU 1 only (GPU 2 stays free for the
# user; GPU 1 is released by konly right before this fires).
set -uo pipefail
cd /home/sjkim/MNPO
LOGDIR=/ext_hdd/sjkim/mnpo/logs
OUTROOT=/ext_hdd/sjkim/mnpo/outputs_ronpo_h200

while pgrep -f "run_mnpo.*os_ronpo_konly_k05" >/dev/null; do
  sleep 120
done
if [[ ! -f "$OUTROOT/os_ronpo_konly_k05/all_results.json" ]]; then
  echo "[chain] konly arm missing all_results.json; aborting" >&2
  exit 1
fi
while [[ ! -f "$OUTROOT/os_ronpo_aonly_k05_b200/all_results.json" ]]; do
  sleep 300
done

echo "[chain] konly done and aonly-b200 transferred at $(date -Is); running factorized eval"
NEW_MODELS="konly=$OUTROOT/os_ronpo_konly_k05 aonly=$OUTROOT/os_ronpo_aonly_k05_b200" \
WORK_DIR=/ext_hdd/sjkim/mnpo/eval/ronpo_factorized_stage2_20260720 \
GPUS="1" \
bash scripts/revision/eval_factorized_arms.sh > "$LOGDIR/eval_factorized_20260720.log" 2>&1
echo "[chain] factorized eval exit=$? at $(date -Is)"
