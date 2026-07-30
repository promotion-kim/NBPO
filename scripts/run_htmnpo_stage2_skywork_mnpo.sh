#!/usr/bin/env bash
set -euo pipefail

cd /home/sjkim/MNPO

export CUDA_VISIBLE_DEVICES=1,2
export NUM_PROCESSES=2
export DECODE_GPUS=2
export RM_GPUS=1,2
export RM_PARALLEL=1
export STAGES=2
export RUN_HTMNPO=1
export RUN_RONPO=0
export RUN_INPO=0
export RUN_SPPO=0
export PLAYERS=skywork
export HT_HISTORY_PLAYERS=skywork,athene,armo
export HT_HISTORY_PATHS_STAGE_2=/ext_hdd/sjkim/mnpo/ht_stage1_out/qwen2.5-1.5b-instruct_htmnpo_athene_online_multiobj_stage_1,/ext_hdd/sjkim/mnpo/hf_models/htmnpo-armorm-qwen25-1p5b-stage1
export OUTPUT_ROOT=/ext_hdd/sjkim/mnpo/ht_stage1_out
export WORK_ROOT=/ext_hdd/sjkim/mnpo/data/qwen2.5-1.5b-instruct_online_htmnpo_ronpo
export ACCELERATE_CONFIG=/home/sjkim/MNPO/accelerate_configs/deepspeed_zero3.yaml
export TRAIN_PER_DEVICE_BATCH_SIZE=8
export TRAIN_PER_DEVICE_EVAL_BATCH_SIZE=8
export TRAIN_GRADIENT_ACCUMULATION_STEPS=1
export PRECOMPUTE_BATCH_SIZE=8
export TRAIN_GENERATE_DURING_EVAL=false
export TRAIN_SAVE_TOTAL_LIMIT=3
export TRAIN_SAVE_STEPS=100
export TRAIN_LOGGING_STEPS=5
export FORCE_PRECOMPUTE=0
export FORCE_SCORE_SKYWORK=0
export DECODE_BATCH_SIZE=512
export RM_BATCH_SIZE=1
export RM_SAMPLE_BATCH_SIZE=8
export RM_MAX_SEQ_LENGTH=4096
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /ext_hdd/sjkim/mnpo/logs

for model_path in /ext_hdd/sjkim/mnpo/ht_stage1_out/qwen2.5-1.5b-instruct_htmnpo_skywork_online_multiobj_stage_1 ${HT_HISTORY_PATHS_STAGE_2//,/ }; do
  if [[ ! -f "$model_path/config.json" ]]; then
    echo "Missing model config: $model_path/config.json" >&2
    exit 1
  fi
  if [[ ! -f "$model_path/model.safetensors" ]]; then
    echo "Missing model weights: $model_path/model.safetensors" >&2
    exit 1
  fi
done

bash run_qwen_online_htmnpo_ronpo.sh 2>&1 | tee /ext_hdd/sjkim/mnpo/logs/htmnpo_stage2_skywork_mnpo_$(date +%Y%m%d_%H%M%S).log
