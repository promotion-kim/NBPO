#!/usr/bin/env bash
# Register the fixed Stage-4 host-local queues after the all-arm stability lock is present.
set -euo pipefail
[[ $# -eq 6 ]] || { echo "usage: $0 PROJECT VENV STAGE3_EXP STAGE4_EXP HOST GPU_LIST" >&2; exit 2; }
PROJECT=$1; VENV=$2; E3=$3; E4=$4; HOST=$5; GPUS=$6
R=$PROJECT/analysis/p10_saferlhf_training_seed43_20260718
LOCK=$E4/continuation_lock.json
mkdir -p "$E4/logs/stage4" "$E4/logs/queues"
while [[ ! -f "$LOCK" ]]; do sleep 30; done
queue_arm() {
  local arm=$1 loss=$2 target=$3 gpu=$4 gate_now=${5:-1} parent=$E3/stage3/$arm/train/full
  if [[ -s "$E4/early_locks/$arm.json" || -s "$E4/stage4_stability_p8_locked_panel/gates/$arm.json" ]]; then
    echo "$(date -Is) $arm is owned by an early Stage-4 launch; skipping duplicate queue" >> "$E4/logs/queues/launch_${HOST}.log"
    return
  fi
  nohup "$R/wait_for_continuation_pool.sh" "$PROJECT" "$E4" stage4 "$arm" "$parent" "$gpu" "$gpu" "$E4/logs/stage4/queue_pool_${arm}_${HOST}g${gpu}.log" > "$E4/logs/queues/queue_pool_${arm}_${HOST}g${gpu}.pidlog" 2>&1 < /dev/null &
  echo $! > "$E4/logs/queues/queue_pool_${arm}_${HOST}g${gpu}.pid"
  nohup "$R/wait_for_continuation_train.sh" "$PROJECT" "$VENV" "$E4" stage4 "$arm" "$parent" "$loss" "$target" "$gpu" "$E4/logs/stage4/queue_train_${arm}_${HOST}g${gpu}.log" > "$E4/logs/queues/queue_train_${arm}_${HOST}g${gpu}.pidlog" 2>&1 < /dev/null &
  echo $! > "$E4/logs/queues/queue_train_${arm}_${HOST}g${gpu}.pid"
  if [[ $gate_now -eq 1 ]]; then
    nohup "$R/wait_for_continuation_gate.sh" "$PROJECT" "$E4" stage4 "$arm" "$gpu" "$E4/logs/stage4_gates/queue_${arm}_${HOST}g${gpu}.log" > "$E4/logs/queues/queue_gate_${arm}_${HOST}g${gpu}.pidlog" 2>&1 < /dev/null &
    echo $! > "$E4/logs/queues/queue_gate_${arm}_${HOST}g${gpu}.pid"
  fi
}
queue_after() {
  local after=$1 arm=$2 loss=$3 target=$4 gpu=$5 parent=$E3/stage3/$arm/train/full
  nohup bash -c "while [ ! -f \"$E4/stage4/$after/train/full/job_status.json\" ]; do sleep 20; done; exec \"$R/wait_for_continuation_pool.sh\" \"$PROJECT\" \"$E4\" stage4 \"$arm\" \"$parent\" \"$gpu\" \"$gpu\" \"$E4/logs/stage4/queue_pool_${arm}_${HOST}g${gpu}.log\"" > "$E4/logs/queues/queue_pool_${arm}_${HOST}g${gpu}.pidlog" 2>&1 < /dev/null &
  echo $! > "$E4/logs/queues/queue_pool_${arm}_${HOST}g${gpu}.pid"
  nohup "$R/wait_for_continuation_train.sh" "$PROJECT" "$VENV" "$E4" stage4 "$arm" "$parent" "$loss" "$target" "$gpu" "$E4/logs/stage4/queue_train_${arm}_${HOST}g${gpu}.log" > "$E4/logs/queues/queue_train_${arm}_${HOST}g${gpu}.pidlog" 2>&1 < /dev/null &
  echo $! > "$E4/logs/queues/queue_train_${arm}_${HOST}g${gpu}.pid"
  nohup "$R/wait_for_continuation_gate.sh" "$PROJECT" "$E4" stage4 "$arm" "$gpu" "$E4/logs/stage4_gates/queue_${arm}_${HOST}g${gpu}.log" > "$E4/logs/queues/queue_gate_${arm}_${HOST}g${gpu}.pidlog" 2>&1 < /dev/null &
  echo $! > "$E4/logs/queues/queue_gate_${arm}_${HOST}g${gpu}.pid"
}
queue_gate_after() {
  local after=$1 arm=$2 gpu=$3
  nohup "$R/retry_gate_after.sh" "$PROJECT" "$E4" stage4 "$arm" "$gpu" "$after" \
    "$E4/logs/stage4_gates/queue_${arm}_after_${after}_${HOST}g${gpu}.log" \
    > "$E4/logs/queues/queue_gate_${arm}_${HOST}g${gpu}.pidlog" 2>&1 < /dev/null &
  echo $! > "$E4/logs/queues/queue_gate_${arm}_${HOST}g${gpu}.pid"
}
if [[ "$HOST" == h1 ]]; then
  IFS=, read -r g0 g1 g2 g3 <<< "$GPUS"
  queue_arm ronpo_os ronpo target_os_k0p1 "$g0" 0
  queue_arm ronpo_topmass ronpo target_topmass_k0p1 "$g1" 0
  queue_arm inpo_avg inpo - "$g2" 0
  queue_arm simpo simpo - "$g3"
  queue_after ronpo_os ipo ipo - "$g0"
  queue_after ronpo_topmass ht_mnpo_harmless ht_mnpo ht_target "$g1"
  queue_after inpo_avg ht_mnpo_helpfulness ht_mnpo ht_target_helpfulness "$g2"
  queue_gate_after ipo ronpo_os "$g0"
  queue_gate_after ht_mnpo_harmless ronpo_topmass "$g1"
  queue_gate_after ht_mnpo_helpfulness inpo_avg "$g2"
elif [[ "$HOST" == h2 ]]; then
  IFS=, read -r g0 g1 <<< "$GPUS"
  queue_arm dpo dpo - "$g0"
  queue_arm sppo_avg sppo - "$g1"
else
  echo "unsupported host: $HOST" >&2
  exit 2
fi
echo "$(date -Is) Stage-4 queues registered for $HOST" >> "$E4/logs/queues/launch_${HOST}.log"
