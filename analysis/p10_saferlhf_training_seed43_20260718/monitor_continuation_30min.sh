#!/usr/bin/env bash
# Read-only periodic evidence log. Launch queues independently enforce three fresh idle samples.
set -euo pipefail
[[ $# -eq 3 ]] || { echo "usage: $0 EXP STAGE HOST" >&2; exit 2; }
EXP=$1; STAGE=$2; HOST=$3
OUT=$EXP/hourly
mkdir -p "$OUT"
while :; do
  stamp=$(date +%Y%m%dT%H%M%S%z)
  {
    printf '{\n  "timestamp_kst": "%s",\n  "host": "%s",\n  "stage": "%s",\n' "$(date -Is)" "$HOST" "$STAGE"
    printf '  "gpu_snapshot": '
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | python3 -c 'import json,sys; print(json.dumps([x.strip() for x in sys.stdin if x.strip()]))'
    printf '  ,"compute_processes": '
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | python3 -c 'import json,sys; print(json.dumps([x.strip() for x in sys.stdin if x.strip()]))'
    printf '  ,"completed_full_arms": '
    find "$EXP/$STAGE" -path '*/train/full/job_status.json' -print 2>/dev/null | sort | python3 -c 'import json,sys; print(json.dumps([x.strip() for x in sys.stdin if x.strip()]))'
    printf '}\n'
  } > "$OUT/${stamp}_${HOST}.json"
  sleep 1800
done
