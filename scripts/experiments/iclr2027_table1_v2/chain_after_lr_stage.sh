#!/usr/bin/env bash
# Wait for the LR-stage arms, score them, then run the Rao-Blackwell diagnostic.
#
# Chained rather than launched by hand so the GPUs are never idle between
# stages. Each step logs to its own file; the chain stops at the first failure
# instead of running the next stage on a missing input.
set -euo pipefail

S=/work/iclr27_table1_v2/smoke/train2
T=/work/iclr27_table1_v2/smoke/train3
L=/work/iclr27_table1_v2/logs
PARENT=/work/models/bases/Llama-3.1-8B-Instruct
cd /work/iclr27_table1_v2/code

echo "[chain] waiting for the LR-stage arms"
until [ -f $S/arms/lr2em7/trainer_state.json ] \
   && [ -f $S/arms/lr1em6/trainer_state.json ] \
   && [ -f $S/arms/DIAGlr1em5/trainer_state.json ]; do sleep 30; done
# the weights land a little after trainer_state
until [ -f $S/arms/lr2em7/model.safetensors ] \
   && [ -f $S/arms/lr1em6/model.safetensors ] \
   && [ -f $S/arms/DIAGlr1em5/model.safetensors ]; do sleep 15; done
echo "[chain] LR stage done, scoring"

i=0
for a in lr2em7 lr1em6 DIAGlr1em5; do
  bash scripts/experiments/iclr2027_table1_v2/score_candidate.sh $S/arms/$a 1.0 $a $i \
    > $L/score_$a.log 2>&1 &
  i=$((i+1))
done
wait
echo "[chain] scoring done"

echo "[chain] Rao-Blackwell precompute"
CUDA_VISIBLE_DEVICES=0 HF_HOME=/root/hf_cache python3 -u -m mnpo_scripts.precompute \
  --model_name_or_path $PARENT --ref_model $PARENT --history_paths $PARENT \
  --train_dir $T/pairs_nbpo/pairs_train.jsonl \
  --test_dir $T/pairs_nbpo/pairs_test.jsonl \
  --output_dir $T/precomputed \
  --solver_artifact_path $T/solver_nbpo/solution.json \
  --logp_reduction sum --truncation_mode keep_end --ronpo_target_mode none \
  --apply_chat_template true --max_length 2048 --max_prompt_length 1024 \
  --per_device_train_batch_size 2 > $L/precompute_rb.log 2>&1

echo "[chain] Rao-Blackwell training"
python3 scripts/experiments/iclr2027_table1_v2/launch_smoke_arm.py \
  --pairs-dir $T/pairs_nbpo --solver-dir $T/solver_nbpo --precomputed $T/precomputed \
  --parent $PARENT --out-dir $T/arms/DIAGrb --run-config $T/arms/DIAGrb_run_config.yaml \
  --eta 1.0 --learning-rate 5e-7 --max-steps 300 > $L/rb_config.log 2>&1
CUDA_VISIBLE_DEVICES=0 HF_HOME=/root/hf_cache python3 -m accelerate.commands.launch \
  --num_processes 1 --main_process_port 29520 -m mnpo_scripts.run_mnpo \
  $T/arms/DIAGrb_run_config.yaml > $L/train_DIAGrb.log 2>&1

echo "[chain] Rao-Blackwell scoring"
sed 's#/smoke/train2#/smoke/train3#g' \
  scripts/experiments/iclr2027_table1_v2/score_candidate.sh > /tmp/score_rb.sh
bash /tmp/score_rb.sh $T/arms/DIAGrb 1.0 DIAGrb 0 > $L/score_DIAGrb.log 2>&1
echo "[chain] COMPLETE"
