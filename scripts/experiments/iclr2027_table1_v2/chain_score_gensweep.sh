#!/usr/bin/env bash
# Score each arm of the prompt-count sweep as soon as its weights land.
#
# All four are scored against the SAME held-out split with the SAME canonical
# target, so the only thing separating them is how many training prompts their
# target came from. Each arm is waited on independently: the N=8 arm finishes no
# earlier than the others here (equal update budget), but a crash in one must not
# hold the other three.
set -uo pipefail

C=/work/iclr27_table1_v2/smoke/canon
L=/work/iclr27_table1_v2/logs
cd /work/iclr27_table1_v2/code
export HF_HOME=/work/hf_cache
sed 's#/smoke/train2#/smoke/canon#g' \
  scripts/experiments/iclr2027_table1_v2/score_candidate.sh > /tmp/score_canon.sh

score_when_ready () {
  local arm="$1" gpu="$2"
  until [ -f "$C/arms/$arm/model.safetensors" ]; do sleep 45; done
  sleep 20
  echo "[score] $arm -> gpu $gpu"
  bash /tmp/score_canon.sh "$C/arms/$arm" 1.0 "$arm" "$gpu" > "$L/score_$arm.log" 2>&1
  echo "[score] $arm DONE"
}

score_when_ready canonN8   0 &
score_when_ready canonN50  1 &
score_when_ready canonN200 2 &
score_when_ready canonN700 3 &
wait
echo "[score] ALL SCORED"
