#!/usr/bin/env bash
# Score each Rao-Blackwell probe the moment its training finishes.
#
# The clip probe finishes about ninety minutes before the 1200-step probe, so
# they are waited on independently rather than as a batch: the first answer
# should not sit idle behind the second.
set -uo pipefail

T=/work/iclr27_table1_v2/smoke/train3
L=/work/iclr27_table1_v2/logs
cd /work/iclr27_table1_v2/code
sed 's#/smoke/train2#/smoke/train3#g' \
  scripts/experiments/iclr2027_table1_v2/score_candidate.sh > /tmp/score_rb.sh

score_when_ready () {
  local arm="$1" gpu="$2"
  until [ -f "$T/arms/$arm/model.safetensors" ]; do sleep 30; done
  sleep 20                       # let the write settle
  echo "[probe] $arm trained, scoring on gpu $gpu"
  bash /tmp/score_rb.sh "$T/arms/$arm" 1.0 "$arm" "$gpu" > "$L/score_$arm.log" 2>&1
  echo "[probe] $arm SCORED"
}

score_when_ready DIAGrb_clip100 0 &
score_when_ready DIAGrb_steps1200 0 &
wait
echo "[probe] ALL DONE"
