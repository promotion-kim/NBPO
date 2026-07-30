#!/usr/bin/env bash
# Wait until a GPU has been continuously idle (<1000 MiB) for STABLE_MIN
# minutes, then launch one B200 arm on it. Used to backfill polymer GPUs as
# the avg stage-2 pipelines shrink from scoring (2 GPUs) to training (1 GPU).
#   GPU=1 TARGET_COL=... NAME=... SEED=43 STABLE_MIN=20 bash wait_free_gpu_then_arm.sh
set -uo pipefail
GPU="${GPU:?}"; STABLE_MIN="${STABLE_MIN:-20}"
SJ=/NHNHOME/AIPR/sjkim
stable=0
while [[ "$stable" -lt "$STABLE_MIN" ]]; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ')
  if [[ -n "$used" && "$used" -lt 1000 ]]; then
    stable=$((stable + 1))
  else
    stable=0
  fi
  sleep 60
done
echo "[waiter] GPU $GPU idle for ${STABLE_MIN}m at $(date -Is); launching $NAME"
exec bash "$SJ/MNPO_rev_20260720/scripts/revision/run_b200_ronpo_arm.sh"
