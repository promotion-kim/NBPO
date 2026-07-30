#!/usr/bin/env bash
# Resume RONPO-on-gemma-2-2b-it (5 ArmoRM heads) from the ALREADY-DECODED pool:
# merge seeds -> score 5 ArmoRM heads (sharded x4) -> build RONPO pairs ->
# precompute -> train. Runs on the ronpo host (4 B200). wandb on.
set -uo pipefail
SJ=/NHNHOME/AIPR/sjkim
PROJECT=$SJ/MNPO_rev_20260720
WORK=$SJ/ronpo_gemma_20260720
PY=$SJ/venv_clean/bin/python
CACHE=$SJ/baseline_repair_1p5b_20260714/cache/huggingface/hub
export PYTHONPATH=$PROJECT HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface
export HF_HUB_CACHE=$CACHE HUGGINGFACE_HUB_CACHE=$CACHE TOKENIZERS_PARALLELISM=false
export WANDB_ENTITY=promotion-kim WANDB_PROJECT=mnpo WANDB_MODE=online WANDB_DIR=$WORK
export MNPO_DISABLE_APEX=1 NCCL_SOCKET_IFNAME=lo
IDX=6,7,8,9,10
NAMES="instruction_following truthfulness honesty helpfulness safety"
NAMES_CSV=instruction_following,truthfulness,honesty,helpfulness,safety
BASE=google/gemma-2-2b-it
mkdir -p $WORK/pool/logs $WORK/scored $WORK/pairs

for sp in train test; do
  d=$WORK/pool/$sp
  mkdir -p $d/merged
  cp -f $d/gpu*/output_*.json $d/merged/ 2>/dev/null || true
  if [[ ! -s $d/merged/all_outputs.json ]]; then
    $PY -m on_policy_data_gen.post_process --generation_file_dir $d/merged
  fi
done

for sp in train test; do
  if [[ ! -s $WORK/scored/${sp}_safety.jsonl ]]; then
    pids=()
    for s in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$s $PY -m on_policy_data_gen.rm_armo_multihead \
        --input_file $WORK/pool/$sp/merged/all_outputs.json --output_dir $WORK/scored --split $sp \
        --indices $IDX --names $NAMES_CSV --cache_dir $CACHE --batch_size 8 --sample_batch_size 32 \
        --num_shards 4 --shard_index $s > $WORK/pool/logs/score_${sp}_sh${s}.log 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    for n in $NAMES; do cat $WORK/scored/${sp}_${n}.jsonl.shard* > $WORK/scored/${sp}_${n}.jsonl; done
  fi
done

sfiles_tr=""; sfiles_te=""
for n in $NAMES; do
  sfiles_tr="$sfiles_tr ${n}=$WORK/scored/train_${n}.jsonl"
  sfiles_te="$sfiles_te ${n}=$WORK/scored/test_${n}.jsonl"
done
if [[ ! -s $WORK/pairs/train_ronpo.jsonl ]]; then
  $PY -m mnpo_scripts.build_multi_objective_dataset --scored_files $sfiles_tr \
    --mnpo_output $WORK/pairs/train_mnpo_unused.jsonl --ronpo_output $WORK/pairs/train_ronpo.jsonl \
    --merged_output $WORK/pairs/train_merged.jsonl --normalization minmax --ronpo_pair_strategy sigma \
    --ronpo_policy_pair_mode expected_relative_policy_vs_policy --ronpo_policy_samples_per_atom 1 \
    --adversary_steps 25 --adversary_alpha 1.0 --adversary_kappa 0.05 --preference_scale 8.0 --pairs_per_prompt 2
  $PY -m mnpo_scripts.build_multi_objective_dataset --scored_files $sfiles_te \
    --mnpo_output $WORK/pairs/test_mnpo_unused.jsonl --ronpo_output $WORK/pairs/test_ronpo.jsonl \
    --merged_output $WORK/pairs/test_merged.jsonl --normalization minmax --ronpo_pair_strategy sigma \
    --ronpo_policy_pair_mode expected_relative_policy_vs_policy --ronpo_policy_samples_per_atom 1 \
    --adversary_steps 25 --adversary_alpha 1.0 --adversary_kappa 0.05 --preference_scale 8.0 --pairs_per_prompt 2
fi

if [[ ! -d $WORK/precomputed ]]; then
  $PY -m accelerate.commands.launch --num_processes=1 -m mnpo_scripts.precompute \
    --model_name_or_path $BASE --ref_model $BASE --history_paths $BASE \
    --train_dir $WORK/pairs/train_ronpo.jsonl --test_dir $WORK/pairs/test_ronpo.jsonl \
    --output_dir $WORK/precomputed --per_device_train_batch_size 8 --max_length 2048 --max_prompt_length 1800 \
    --apply_chat_template true --auto_insert_empty_system_msg false --ronpo_target_mode none --sanity_check False \
    > $WORK/pool/logs/precompute.log 2>&1
fi

$PY -m accelerate.commands.launch --config_file $PROJECT/accelerate_configs/single_gpu.yaml \
  --num_processes=1 --main_process_port=29850 -m mnpo_scripts.run_mnpo \
  $PROJECT/training_configs/ronpo/qwen2.5-1.5b-instruct-ronpo-multiobj-iter1.yaml \
  --model_name_or_path=$BASE --dataset_mixer=$WORK/precomputed:1.0 \
  --loss_type=ronpo --ronpo_alpha=1.0 --ronpo_tau=0.05 --ronpo_target_column=ronpo_target \
  --max_history_t=1 --history_weights=1.0 --learning_rate=5.0e-7 --warmup_ratio=0.1 --num_train_epochs=1 \
  --per_device_train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing=true \
  --max_length=2048 --max_prompt_length=1800 --do_eval=false --eval_strategy=no --generate_during_eval=false \
  --save_strategy=steps --save_steps=800 --save_total_limit=1 --logging_steps=10 \
  --output_dir=$WORK/ronpo_gemma_5obj --run_name=ronpo-gemma-2b-5obj \
  2>&1 | tee $WORK/pool/logs/train.log
echo "[ronpo-gemma] done at $(date -Is)"
