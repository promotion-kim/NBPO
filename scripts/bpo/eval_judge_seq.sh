#!/usr/bin/env bash
# Judge all 3 stage-2 arms sequentially on 4 local GPUs, then per-judge surplus.
# Usage: eval_judge_seq.sh ROOT
set -euo pipefail
ROOT=$1
P=/NHNHOME/AIPR/sjkim/MNPO_rev_20260710
V=/NHNHOME/AIPR/sjkim/venv_clean
ED=$ROOT/stage2/eval
for arm in unif nbs ks; do
  bash $P/scripts/bpo/judge_gen.sh $ED/$arm/gens $ED/$arm/verdicts 0 4
  sleep 8
  while tmux ls 2>/dev/null | grep -q '^jg'; do sleep 15; done
  cat $ED/$arm/verdicts/shard*.jsonl > $ED/$arm/verdicts.jsonl
  $V/bin/python $P/scripts/bpo/eval_bpo_surplus.py \
    --verdicts $ED/$arm/verdicts.jsonl --label $arm --out $ED/$arm/surplus.json
  echo "ARM_DONE $arm"
done
echo ALL_EVAL_DONE
