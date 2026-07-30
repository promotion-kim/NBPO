#!/usr/bin/env bash
set -euo pipefail

SJ=/NHNHOME/AIPR/sjkim
PROJECT=$SJ/MNPO_rev_20260720
OUT=$SJ/rmod_20260720/saferlhf_ksweep_chat_b16_n1000
RUNNER=$PROJECT/scripts/revision/rmod/run_rmod_saferlhf_ksweep_chat.sh
ROLE=${ROLE:?shard0, shard1, or merge}
GPU=${GPU:-0}

sample_gpu() {
  for _ in 1 2 3; do
    date -Is >> "$OUT/logs/gpu${GPU}_${ROLE}_prelaunch.txt"
    nvidia-smi -i "$GPU" --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits >> "$OUT/logs/gpu${GPU}_${ROLE}_prelaunch.txt"
    sleep 2
  done
}

case "$ROLE" in
  shard0)
    while [[ ! -s $SJ/rmod_20260720/saferlhf_ksweep/k16_summary.json ]]; do sleep 60; done
    sample_gpu
    GPU=$GPU K=1 BLOCK=16 NPROMPTS=1000 bash "$RUNNER"
    GPU=$GPU K=16 BLOCK=16 NPROMPTS=1000 SHARD_INDEX=0 SHARD_COUNT=2 bash "$RUNNER"
    ;;
  shard1)
    until python3 - "$OUT/k4_summary.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
raise SystemExit(not (p.exists() and "saferlhf_vf_disjoint_run" in
                      json.loads(p.read_text()).get("value_function_checkpoint", "")))
PY
    do sleep 60; done
    sample_gpu
    GPU=$GPU K=16 BLOCK=16 NPROMPTS=1000 SHARD_INDEX=1 SHARD_COUNT=2 bash "$RUNNER"
    ;;
  merge)
    while [[ ! -s $OUT/k16_shard0of2_summary.json || ! -s $OUT/k16_shard1of2_summary.json ]]; do
      sleep 60
    done
    $SJ/venv_clean/bin/python \
      "$PROJECT/analysis/rmod_validation_20260721/merge_saferlhf_shards.py" \
      --root "$OUT" --k 16 --shards 2 --expected-n 1000 --block-size 16 \
      > "$OUT/logs/merge_k16.log" 2>&1
    ;;
  *)
    echo "unknown ROLE=$ROLE" >&2
    exit 2
    ;;
esac
