#!/usr/bin/env bash
# RONPO-gemma stage 2 (UF 5-obj): pool from the stage-1 policy's own 5-seed
# generations, resume training FROM the stage-1 checkpoint (history = stage-1,
# ref = base gemma), mirroring the paper's 1.5B stage recipe. Then decode the
# UF test prompts and score the 5 ArmoRM heads as split ronpo_gemma_s2.
# Runs on 4 GPUs of one host.  bash ronpo_gemma_stage2.sh
set -uo pipefail
SJ=/NHNHOME/AIPR/sjkim; P=$SJ/MNPO_rev_20260720
S1=$SJ/ronpo_gemma_20260720/kappa_arms/os_k0p05          # stage-1 policy (kappa 0.05)
W=$SJ/ronpo_gemma_s2
CACHE=$SJ/baseline_repair_1p5b_20260714/cache/huggingface/hub
PY=$SJ/venv_clean/bin/python; BASE=google/gemma-2-2b-it
export PYTHONPATH=$P HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface
export HF_HUB_CACHE=$CACHE HUGGINGFACE_HUB_CACHE=$CACHE TOKENIZERS_PARALLELISM=false
export WANDB_ENTITY=promotion-kim WANDB_PROJECT=mnpo WANDB_MODE=online WANDB_DIR=$W
export MNPO_DISABLE_APEX=1 NCCL_SOCKET_IFNAME=lo
IDX=6,7,8,9,10
NAMES_CSV=instruction_following,truthfulness,honesty,helpfulness,safety
mkdir -p $W/pool/logs $W/scored $W/pairs

# 1. stage-2 pool: 5 seeds from the stage-1 policy, split over 4 GPUs
dec() { CUDA_VISIBLE_DEVICES=$1 $PY -u -m on_policy_data_gen.decode \
  --data_dir $P/data/gemma2_ufb_part2_$3.jsonl --model $S1 --seeds $2 \
  --output_dir $W/pool/$3/gpu$1 --num_gpu 1 --temperature 0.8 --top_p 0.95 \
  --max_tokens 1024 --batch_size 512 --dtype bfloat16 --cache_dir $CACHE \
  > $W/pool/logs/dec_$3_gpu$1.log 2>&1; }
dec 0 "13 21" train & p0=$!; dec 1 "42" train & p1=$!; dec 2 "79" train & p2=$!; dec 3 "100" train & p3=$!
wait $p0 $p1 $p2 $p3
dec 0 "13 21" test & q0=$!; dec 1 "42" test & q1=$!; dec 2 "79" test & q2=$!; dec 3 "100" test & q3=$!
wait $q0 $q1 $q2 $q3
for sp in train test; do
  mkdir -p $W/pool/$sp/merged; cp -f $W/pool/$sp/gpu*/output_*.json $W/pool/$sp/merged/
  $PY -m on_policy_data_gen.post_process --generation_file_dir $W/pool/$sp/merged
done

# 2. score 5 heads, sharded x4
for sp in train test; do
  pids=()
  for s in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$s $PY -m on_policy_data_gen.rm_armo_multihead \
      --input_file $W/pool/$sp/merged/all_outputs.json --output_dir $W/scored --split $sp \
      --indices $IDX --names $NAMES_CSV --cache_dir $CACHE --batch_size 8 --sample_batch_size 32 \
      --num_shards 4 --shard_index $s > $W/pool/logs/score_${sp}_sh$s.log 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
  for n in ${NAMES_CSV//,/ }; do cat $W/scored/${sp}_${n}.jsonl.shard* > $W/scored/${sp}_${n}.jsonl; done
done

# 3. pairs + precompute (model/history = stage-1, ref = base)
sf_tr=""; sf_te=""
for n in ${NAMES_CSV//,/ }; do sf_tr="$sf_tr $n=$W/scored/train_${n}.jsonl"; sf_te="$sf_te $n=$W/scored/test_${n}.jsonl"; done
$PY -m mnpo_scripts.build_multi_objective_dataset --scored_files $sf_tr \
  --mnpo_output $W/pairs/train_mnpo_unused.jsonl --ronpo_output $W/pairs/train_ronpo.jsonl \
  --merged_output $W/pairs/train_merged.jsonl --normalization minmax --ronpo_pair_strategy sigma \
  --ronpo_policy_pair_mode expected_relative_policy_vs_policy --ronpo_policy_samples_per_atom 1 \
  --adversary_steps 25 --adversary_alpha 1.0 --adversary_kappa 0.05 --preference_scale 8.0 --pairs_per_prompt 2
$PY -m mnpo_scripts.build_multi_objective_dataset --scored_files $sf_te \
  --mnpo_output $W/pairs/test_mnpo_unused.jsonl --ronpo_output $W/pairs/test_ronpo.jsonl \
  --merged_output $W/pairs/test_merged.jsonl --normalization minmax --ronpo_pair_strategy sigma \
  --ronpo_policy_pair_mode expected_relative_policy_vs_policy --ronpo_policy_samples_per_atom 1 \
  --adversary_steps 25 --adversary_alpha 1.0 --adversary_kappa 0.05 --preference_scale 8.0 --pairs_per_prompt 2
$PY -m accelerate.commands.launch --num_processes=4 -m mnpo_scripts.precompute \
  --model_name_or_path $S1 --ref_model $BASE --history_paths $S1 \
  --train_dir $W/pairs/train_ronpo.jsonl --test_dir $W/pairs/test_ronpo.jsonl \
  --output_dir $W/precomputed --per_device_train_batch_size 8 --max_length 2048 --max_prompt_length 1800 \
  --apply_chat_template true --auto_insert_empty_system_msg false --ronpo_target_mode none --sanity_check False \
  > $W/pool/logs/precompute.log 2>&1

# 4. train stage 2 (init from stage-1)
CUDA_VISIBLE_DEVICES=0 $PY -m accelerate.commands.launch \
  --config_file $P/accelerate_configs/single_gpu.yaml --num_processes=1 --main_process_port=29950 \
  -m mnpo_scripts.run_mnpo $P/training_configs/ronpo/qwen2.5-1.5b-instruct-ronpo-multiobj-iter1.yaml \
  --model_name_or_path=$S1 --dataset_mixer=$W/precomputed:1.0 \
  --loss_type=ronpo --ronpo_alpha=1.0 --ronpo_tau=0.05 --ronpo_target_column=ronpo_target \
  --max_history_t=1 --history_weights=1.0 --learning_rate=5.0e-7 --warmup_ratio=0.1 --num_train_epochs=1 --seed=42 \
  --max_steps=1800 --per_device_train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing=true \
  --max_length=2048 --max_prompt_length=1800 --do_eval=false --eval_strategy=no --generate_during_eval=false \
  --save_strategy=steps --save_steps=800 --save_total_limit=1 --logging_steps=10 \
  --output_dir=$W/stage2 --run_name=ronpo-gemma-2b-s2 > $W/pool/logs/train_s2.log 2>&1
[[ -f $W/stage2/all_results.json ]] || { echo "[s2] TRAIN FAILED"; exit 1; }

# 5. decode UF test @128 + score -> split ronpo_gemma_s2
CUDA_VISIBLE_DEVICES=0 $PY -u -m on_policy_data_gen.decode \
  --data_dir $P/data/gemma2_ufb_part2_test.jsonl --model $W/stage2 --seeds 42 --output_dir $W/stage2/gen \
  --num_gpu 1 --temperature 0.8 --top_p 0.95 --max_tokens 128 --batch_size 256 --dtype bfloat16 --cache_dir $CACHE \
  > $W/pool/logs/dec_s2_eval.log 2>&1
$PY - <<EOF
import json
d=json.load(open("$W/stage2/gen/output_42.json"))
json.dump([{"prompt":r["prompt"],"all_generated_responses":[r["generated_text"]]} for r in d],open("$W/stage2/gen/fmt.json","w"))
EOF
CUDA_VISIBLE_DEVICES=0 $PY -m on_policy_data_gen.rm_armo_multihead \
  --input_file $W/stage2/gen/fmt.json --output_dir $SJ/rmod_20260720/radar/scored --split ronpo_gemma_s2 \
  --indices $IDX --names $NAMES_CSV --cache_dir $CACHE --batch_size 16 --sample_batch_size 32 \
  > $W/pool/logs/score_s2_eval.log 2>&1
echo "[stage2] complete at $(date -Is)"
