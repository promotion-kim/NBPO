#!/usr/bin/env bash
# Stage-2 post-judge: build 3-arm pairs, precompute, train nbs/ks/unif in parallel.
# Usage: stage2_finish.sh ROOT PARENT POLICY_DIR
#   PARENT     = stage-1 policy  (Nash self-play opponent = history, train init)
#   ref_model  = base llama31    (KL anchor mu; run_bpo_stage.sh default BASE, NOT overridden)
set -euo pipefail
ROOT=$1; PARENT=$2; PD=$3
P=/NHNHOME/AIPR/sjkim/MNPO_rev_20260710
V=/NHNHOME/AIPR/sjkim/venv_clean
export P V PYTHONPATH=$P HF_HUB_OFFLINE=1     # do NOT set BASE: precompute ref_model must stay base llama31
SD=$ROOT/stage2
# merge shards -> verdicts
cat $SD/verdicts/shard*.jsonl > $SD/verdicts.jsonl
# build (3 weight rules) + precompute (history/opponent = stage-1 policy; ref = base)
bash $P/scripts/bpo/run_bpo_stage.sh build $ROOT 2 "$PD"
bash $P/scripts/bpo/run_bpo_stage.sh precompute $ROOT 2 "$PARENT" 0
# train 3 arms in parallel on GPU 0,1,2 (init from stage-1 policy)
mkdir -p $SD/train
g=0
for arm in unif nbs ks; do
  tmux new-session -d -s tr_$arm "source /NHNHOME/AIPR/sjkim/.secrets/wandb.env; export P=$P V=$V PYTHONPATH=$P WANDB_ENTITY=promotion-kim WANDB_PROJECT=mnpo; bash $P/scripts/bpo/run_bpo_stage.sh train $ROOT 2 $arm $PARENT $g && echo DONE_$arm > $SD/train/$arm.done"
  g=$((g+1))
done
echo STAGE2_TRAIN_LAUNCHED
