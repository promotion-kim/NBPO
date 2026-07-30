#!/usr/bin/env bash
set -euo pipefail

cd /home/sjkim/MNPO

CHECK_INTERVAL="${CHECK_INTERVAL:-300}"
LOG_DIR="${LOG_DIR:-/ext_hdd/sjkim/mnpo/logs}"
KUBECONFIG_PATH="${KUBECONFIG_PATH:-mlxp/aipr-kubeconfig.yaml}"
H200_JOB="${H200_JOB:-mnpo-ronpo-s2-rel-lr3e8-h200-2g-r2}"
ODIN_SESSIONS="${ODIN_SESSIONS:-ronpo_s2_rel ronpo_s2_lr3e8_g2}"
WATCH_UNTIL="${WATCH_UNTIL:-2026-06-24 23:59:00}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/ronpo_s2_active_watch_$(date +%Y%m%d_%H%M%S).log"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_FILE"
}

deadline_epoch() {
  date -d "$WATCH_UNTIL" +%s
}

now_epoch() {
  date +%s
}

capture_odin_session() {
  local session="$1"
  if tmux has-session -t "=$session" 2>/dev/null; then
    log "odin2 tmux session=$session"
    tmux capture-pane -t "$session" -p -S -35 | tail -35 | tee -a "$LOG_FILE" >/dev/null || true
  else
    log "odin2 tmux session missing: $session"
  fi
}

check_h200_job() {
  log "h200 job status: $H200_JOB"
  kubectl --kubeconfig "$KUBECONFIG_PATH" -n p-aipr get job "$H200_JOB" -o wide | tee -a "$LOG_FILE" >/dev/null || true
  kubectl --kubeconfig "$KUBECONFIG_PATH" -n p-aipr get pods -l "app=$H200_JOB" -o wide | tee -a "$LOG_FILE" >/dev/null || true

  local pod
  pod="$(kubectl --kubeconfig "$KUBECONFIG_PATH" -n p-aipr get pods -l "app=$H200_JOB" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  if [[ -n "$pod" ]]; then
    log "h200 pod tail: $pod"
    kubectl --kubeconfig "$KUBECONFIG_PATH" -n p-aipr logs "$pod" -c main --tail=80 | tee -a "$LOG_FILE" >/dev/null || true
    log "h200 gpu snapshot: $pod"
    kubectl --kubeconfig "$KUBECONFIG_PATH" -n p-aipr exec "$pod" -c main -- nvidia-smi | tee -a "$LOG_FILE" >/dev/null || true
    log "h200 storage snapshot: $pod"
    kubectl --kubeconfig "$KUBECONFIG_PATH" -n p-aipr exec "$pod" -c main -- df -h /data | tee -a "$LOG_FILE" >/dev/null || true
  fi
}

log "active RONPO stage2 watch start until=$WATCH_UNTIL h200_job=$H200_JOB odin_sessions=$ODIN_SESSIONS"

while (( "$(now_epoch)" < "$(deadline_epoch)" )); do
  log "local storage"
  df -h /ext_hdd | tee -a "$LOG_FILE" >/dev/null || true

  log "odin2 gpu snapshot"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits | tee -a "$LOG_FILE" >/dev/null || true

  for session in $ODIN_SESSIONS; do
    capture_odin_session "$session"
  done

  check_h200_job
  sleep "$CHECK_INTERVAL"
done

log "active RONPO stage2 watch reached deadline"
