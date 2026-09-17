#!/usr/bin/env bash
# The prompt-count generalization sweep, on four H200s.
#
# Every arm gets the SAME compute (1200 gradient updates), the SAME canonical
# target, the SAME learning rate and the SAME held-out set. The only thing that
# varies is how many training prompts the target came from. If held-out
# correlation rises with prompt count, the failure was budget; if it falls, the
# per-prompt targets do not transfer and more prompts actively hurt.
#
# These are diagnostics. None of them is a selection candidate and none is a
# final campaign run.
set -uo pipefail

C=/work/iclr27_table1_v2/smoke/canon
L=/work/iclr27_table1_v2/logs
P=/work/models/bases/Llama-3.1-8B-Instruct
cd /work/iclr27_table1_v2/code
export HF_HOME=/work/hf_cache

echo "[gen] waiting for the canonical precompute"
until [ -f $C/precomputed/dataset_dict.json ]; do sleep 30; done
echo "[gen] precompute ready"

for N in 8 50 200; do
  [ -f $C/sub$N/dataset_dict.json ] || \
    python3 scripts/experiments/iclr2027_table1_v2/make_prompt_subset.py \
      --parent $C/precomputed --out $C/sub$N --n-prompts $N
done
echo "[gen] subsets built"

launch () {
  local tag="$1" data="$2" gpu="$3"
  python3 scripts/experiments/iclr2027_table1_v2/launch_smoke_arm.py \
    --pairs-dir $C/pairs_nbpo --solver-dir $C/solver_nbpo --precomputed "$data" \
    --parent $P --out-dir $C/arms/$tag --run-config $C/arms/${tag}_run_config.yaml \
    --eta 1.0 --learning-rate 5e-7 --max-steps 1200 > $L/cfg_$tag.log 2>&1
  sed -i 's/^logging_steps: .*/logging_steps: 25/' $C/arms/${tag}_run_config.yaml
  CUDA_VISIBLE_DEVICES=$gpu nohup python3 -m accelerate.commands.launch \
    --num_processes 1 --main_process_port $((29600 + gpu)) -m mnpo_scripts.run_mnpo \
    $C/arms/${tag}_run_config.yaml > $L/train_$tag.log 2>&1 &
  echo "[gen] launched $tag on gpu $gpu"
}

launch canonN8   $C/sub8        0
launch canonN50  $C/sub50       1
launch canonN200 $C/sub200      2
launch canonN700 $C/precomputed 3
wait
echo "[gen] ALL TRAINED"
