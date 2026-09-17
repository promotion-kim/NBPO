#!/usr/bin/env bash
# The paired old/new reference comparison at 300 updates on the full canonical set.
#
# Both arms share the canonical target, the same solver artifact, the same
# precomputed dataset, the same learning rate, schedule horizon, seed and batch
# order. The ONLY difference is how the proximal centre pi_t is obtained:
#
#   old : the cached history0 columns, grouped into batches by precompute
#   new : a frozen copy of pi_t forwarded through the same collated batch,
#         with the sequence log-probability accumulated in float32
#
# The two numerical fixes travel together because they are one change of
# contract, and the arm is named for that rather than for either half.
set -uo pipefail

C=/work/iclr27_table1_v2/smoke/canon
R=/work/iclr27_table1_v2/smoke/reffix
L=/work/iclr27_table1_v2/logs
P=/work/models/bases/Llama-3.1-8B-Instruct
export HF_HOME=/work/hf_cache

launch () {
  local tag="$1" code="$2" gpu="$3" online="$4"
  cd $code
  PYTHONPATH=$code python3 scripts/experiments/iclr2027_table1_v2/launch_smoke_arm.py \
    --pairs-dir $C/pairs_nbpo --solver-dir $C/solver_nbpo --precomputed $C/precomputed \
    --parent $P --out-dir $R/arms/$tag --run-config $R/arms/${tag}_run_config.yaml \
    --eta 1.0 --learning-rate 5e-7 --max-steps 300 > $L/cfg_$tag.log 2>&1
  sed -i 's/^logging_steps: .*/logging_steps: 25/' $R/arms/${tag}_run_config.yaml
  if [ "$online" = "yes" ]; then
    printf "nbpo_online_reference: true\nnbpo_reference_model_path: %s\n" "$P" \
      >> $R/arms/${tag}_run_config.yaml
  fi
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$code nohup python3 -m accelerate.commands.launch \
    --num_processes 1 --main_process_port $((29560 + gpu)) -m mnpo_scripts.run_mnpo \
    $R/arms/${tag}_run_config.yaml > $L/train_$tag.log 2>&1 &
  echo "[paired] launched $tag on gpu $gpu (online_reference=$online, code=$code)"
}

free_gpu () {   # echo the index of a GPU with no process, or nothing
  for g in 0 1 2 3; do
    if [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; then
      echo $g; return
    fi
  done
}

for tag_spec in "PAIRED300_oldref:/work/iclr27_table1_v2/code_frozen_0131:no" \
                "PAIRED300_newref:/work/iclr27_table1_v2/code_reffix:yes"; do
  tag=${tag_spec%%:*}; rest=${tag_spec#*:}; code=${rest%%:*}; online=${rest##*:}
  g=""
  while [ -z "$g" ]; do g=$(free_gpu); [ -z "$g" ] && sleep 60; done
  launch "$tag" "$code" "$g" "$online"
  sleep 90            # let it claim the GPU before looking for the next free one
done
echo "[paired] both launched"
