#!/usr/bin/env bash
set -uo pipefail

INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
MAX_SNAPSHOTS="${MAX_SNAPSHOTS:-17}"
LOG_FILE="${LOG_FILE:-/home/sjkim/MNPO/results/ronpo_aaai_8h_monitor_20260712.log}"
KEY="/home/sjkim/MNPO/nhn/sjkim-test_key"
PORT="45020"
REMOTE="aipr_lab@127.0.0.1"
KUBECONFIG_PATH="/home/sjkim/MNPO/mlxp/aipr-kubeconfig.yaml"

mkdir -p "$(dirname "$LOG_FILE")"

snapshot() {
  {
    echo "[snapshot-begin] $(date -Is)"
    echo "[b200]"
    ssh -F /dev/null -i "$KEY" -o ConnectTimeout=12 \
      -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no \
      -p "$PORT" "$REMOTE" '
        BASE=/NHNHOME/WORKSPACE/26msit001_A/BASE/aipr_lab_sjkim_eval
        ROOT=$BASE/ronpo_aaai_8h_20260712
        date -Is
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
        ps -u aipr_lab -o pid,etime,stat,args | grep -E "(mnpo_scripts|run_kto|run_simpo|accelerate|vllm.entrypoints|upload_checkpoint)" | grep -v grep || true
        tmux ls 2>/dev/null || true
        if test -d "$ROOT"; then
          find "$ROOT" -maxdepth 3 -type f -name "*.log" -mmin -40 -print | while read -r f; do
            grep -E "Traceback|OutOfMemory|CUDA out of memory|NCCL.*(error|failed)|wandb: ERROR|NaN" "$f" | tail -3 || true
          done
          find "$ROOT" -maxdepth 3 -type f \( -name "status.json" -o -name "run_status.json" \) -print -exec cat {} \; 2>/dev/null || true
        fi
        df -h "$BASE" | tail -1
      ' || echo "[alert] B200 SSH snapshot failed"
    echo "[h200]"
    kubectl --kubeconfig "$KUBECONFIG_PATH" -n p-aipr3 get pods \
      --field-selector=status.phase=Running \
      -o custom-columns='NAME:.metadata.name,NODE:.spec.nodeName,GPU:.spec.containers[*].resources.requests.nvidia\.com/gpu,OWNER:.metadata.ownerReferences[0].name' \
      || echo "[alert] H200 kubectl snapshot failed"
    echo "[snapshot-end] $(date -Is)"
  } >> "$LOG_FILE" 2>&1
}

for ((i = 1; i <= MAX_SNAPSHOTS; i++)); do
  snapshot
  ((i == MAX_SNAPSHOTS)) && break
  sleep "$INTERVAL_SECONDS"
done
