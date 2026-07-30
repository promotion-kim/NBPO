#!/usr/bin/env bash
set -uo pipefail

KUBECONFIG_PATH="${KUBECONFIG_PATH:-/home/sjkim/MNPO/mlxp/aipr-kubeconfig.yaml}"
NAMESPACE="${NAMESPACE:-p-aipr3}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-3600}"
LOG_FILE="${LOG_FILE:-/home/sjkim/MNPO/results/aipr3_h200_hourly_20260712.log}"

snapshot() {
  {
    echo "[snapshot-begin] $(date -Is)"
    kubectl --kubeconfig "$KUBECONFIG_PATH" -n "$NAMESPACE" get pods \
      --field-selector=status.phase=Running \
      -o custom-columns='NAME:.metadata.name,NODE:.spec.nodeName,GPU:.spec.containers[*].resources.requests.nvidia\.com/gpu,OWNER:.metadata.ownerReferences[0].name'
    for job in te-mask-fullrecipe-400k-0708 te-vanilla-fullrecipe-400k-0708; do
      if kubectl --kubeconfig "$KUBECONFIG_PATH" -n "$NAMESPACE" get job "$job" >/dev/null 2>&1; then
        echo "[gpu-job] $job"
        kubectl --kubeconfig "$KUBECONFIG_PATH" -n "$NAMESPACE" exec "job/$job" -c main -- \
          nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader || true
      fi
    done
    echo "[snapshot-end] $(date -Is)"
  } >> "$LOG_FILE" 2>&1
}

while true; do
  snapshot
  [[ "${ONCE:-0}" == "1" ]] && break
  sleep "$INTERVAL_SECONDS"
done
