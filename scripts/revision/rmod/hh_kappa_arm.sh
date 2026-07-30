#!/usr/bin/env bash
# RONPO-gemma-2-2b arm on the HH helpful-vs-harmless CONFLICT at a given adversary
# temperature kappa (OS target), then decode HH test prompts @128 and score the
# helpfulness(9) and safety(10) ArmoRM heads = the (helpful, harmless) plane for
# the Fig-3c-analog Nash-vs-kappa sweep.  KTAG=0p05 GPU=0 bash hh_kappa_arm.sh
set -uo pipefail
KTAG="${KTAG:?}"; GPU="${GPU:-${DISPATCH_GPU:?set GPU or DISPATCH_GPU}}"
SEED="${SEED:-42}"; SFX=""; [[ "$SEED" != "42" ]] && SFX="_s${SEED}"
SJ=/NHNHOME/AIPR/sjkim; P=$SJ/MNPO_rev_20260720; W=$SJ/ronpo_hh_20260721
CACHE=$SJ/baseline_repair_1p5b_20260714/cache/huggingface/hub
PY=$SJ/venv_clean/bin/python; BASE=google/gemma-2-2b-it
export PYTHONPATH=$P HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface
export HF_HUB_CACHE=$CACHE HUGGINGFACE_HUB_CACHE=$CACHE
export WANDB_ENTITY=promotion-kim WANDB_PROJECT=mnpo WANDB_MODE=online WANDB_DIR=$W
export TOKENIZERS_PARALLELISM=false MNPO_DISABLE_APEX=1 NCCL_SOCKET_IFNAME=lo
OUT=$W/kappa_arms/os_k${KTAG}${SFX}; SPLIT=hh_ronpo_k${KTAG}${SFX}
mkdir -p $W/kappa_arms/logs $OUT
[[ -s $SJ/rmod_20260720/radar/scored/${SPLIT}_safety.jsonl ]] && { echo "[hh $SPLIT] already scored"; exit 0; }

CUDA_VISIBLE_DEVICES=$GPU $PY -m accelerate.commands.launch \
  --config_file $P/accelerate_configs/single_gpu.yaml --num_processes=1 --main_process_port=$((29900+GPU)) \
  -m mnpo_scripts.run_mnpo $P/training_configs/ronpo/qwen2.5-1.5b-instruct-ronpo-multiobj-iter1.yaml \
  --model_name_or_path=$BASE --dataset_mixer=${DATASET:-$W/precomputed_kappa}:1.0 \
  --loss_type=ronpo --ronpo_alpha=1.0 --ronpo_tau=0.05 --ronpo_target_column=target_os_k${KTAG} \
  --max_history_t=1 --history_weights=1.0 --learning_rate=5.0e-7 --warmup_ratio=0.1 --num_train_epochs=1 --seed=$SEED \
  --max_steps=${MAX_STEPS:-1300} \
  --per_device_train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing=true \
  --max_length=2048 --max_prompt_length=1800 --do_eval=false --eval_strategy=no --generate_during_eval=false \
  --save_strategy=steps --save_steps=800 --save_total_limit=1 --logging_steps=10 \
  --output_dir=$OUT --run_name=ronpo-hh-2b-os-k${KTAG}${SFX} > $W/kappa_arms/logs/train_k${KTAG}${SFX}.log 2>&1
[[ -f $OUT/all_results.json ]] || { echo "[hh k$KTAG] TRAIN FAILED"; exit 1; }

CUDA_VISIBLE_DEVICES=$GPU $PY -u -m on_policy_data_gen.decode \
  --data_dir $W/hh_test_prompts.json --model $OUT --seeds 42 --output_dir $OUT/gen --num_gpu 1 \
  --temperature 0.8 --top_p 0.95 --max_tokens 128 --batch_size 256 --dtype bfloat16 --cache_dir $CACHE \
  > $W/kappa_arms/logs/decode_k${KTAG}${SFX}.log 2>&1
$PY -c "
import json; d=json.load(open('$OUT/gen/output_42.json'))
json.dump([{'prompt':r['prompt'],'all_generated_responses':[r['generated_text']]} for r in d],open('$OUT/gen/fmt.json','w'))"

CUDA_VISIBLE_DEVICES=$GPU $PY -m on_policy_data_gen.rm_armo_multihead \
  --input_file $OUT/gen/fmt.json --output_dir $SJ/rmod_20260720/radar/scored --split $SPLIT \
  --indices 9,10 --names helpfulness,safety \
  --cache_dir $CACHE --batch_size 16 --sample_batch_size 32 > $W/kappa_arms/logs/score_k${KTAG}${SFX}.log 2>&1
echo "[hh $SPLIT] done at $(date -Is)"
