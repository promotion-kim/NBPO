#!/bin/bash
# Mechanism controls, one 4-GPU arm at a time, ordered by how far the control's
# teacher actually sits from the NBPO-Nash teacher on dev (mean total variation):
#   bt_rm_nash 0.0485  >  fixed_reference 0.0281  >  maxmin(L1-matched) 0.00003
# A later arm never starts until the previous one has recorded an exit.
set -u
ROOT=/work/nbpo_repair_20260909
export PYTHONPATH=$ROOT/deps_train:$ROOT/code
cd $ROOT/code || exit 1
for arm in btrm_mse_s42 fixedref_mse_s42 maxmin_l1m_mse_s42; do
  if [ -f "$ROOT/jobs/$arm/exit.json" ]; then
    echo "$(date -u +%FT%TZ) $arm already has an exit record; skipping"
    continue
  fi
  echo "$(date -u +%FT%TZ) starting $arm"
  python3 -m scripts.experiments.nbpo_repair_20260909.train_job \
      --root $ROOT --config $ROOT/configs/$arm.yaml --job $arm \
      > $ROOT/logs/job_$arm.log 2>&1
  echo "$(date -u +%FT%TZ) $arm exit=$? $(cat $ROOT/jobs/$arm/exit.json 2>/dev/null | tr -d '\n')"
done
echo "$(date -u +%FT%TZ) mechanism chain done"
