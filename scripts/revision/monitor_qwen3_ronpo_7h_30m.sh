#!/usr/bin/env bash
set -uo pipefail

INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
MAX_SNAPSHOTS="${MAX_SNAPSHOTS:-15}"
LOG_FILE="${LOG_FILE:-/home/sjkim/MNPO/results/qwen3_ronpo_7h_monitor_20260712.log}"
KEY="/home/sjkim/MNPO/nhn/sjkim-test_key"
PORT="45020"
REMOTE="aipr_lab@127.0.0.1"
KUBECONFIG_PATH="/home/sjkim/MNPO/mlxp/aipr-kubeconfig.yaml"
NAMESPACE="p-aipr3"

mkdir -p "$(dirname "$LOG_FILE")"

snapshot() {
  {
    echo "[snapshot-begin] $(date -Is)"
    echo "[b200]"
    ssh -F /dev/null -i "$KEY" \
      -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no \
      -o ConnectTimeout=15 -p "$PORT" "$REMOTE" '
        BASE=/NHNHOME/WORKSPACE/26msit001_A/BASE/aipr_lab_sjkim_eval
        ROOT=$BASE/revision_qwen3_8b/full_iter1/retrain/ronpo-stage2-top1-sft-20260712-0912
        date -Is
        cat "$ROOT/status.json" 2>/dev/null || true
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
        ps -u aipr_lab -o pid,etime,cmd | grep -E "(mnpo_scripts.precompute|mnpo_scripts.run_mnpo|run_qwen3_localrm_eval|vllm.entrypoints)" | grep -v grep || true
        for f in "$ROOT"/logs/*.log; do
          test -f "$f" || continue
          grep -E "Traceback|OutOfMemory|CUDA out of memory|NCCL.*(error|failed)|wandb: ERROR" "$f" | tail -3 || true
        done
        df -h "$BASE" | tail -1
      ' || echo "[alert] B200 SSH snapshot failed"
    echo "[h200]"
    kubectl --kubeconfig "$KUBECONFIG_PATH" -n "$NAMESPACE" get pods \
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
