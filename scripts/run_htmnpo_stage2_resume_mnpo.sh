#!/usr/bin/env bash
set -euo pipefail

cd /home/sjkim/MNPO

export CUDA_VISIBLE_DEVICES=0,1
export NUM_PROCESSES=2
export DECODE_GPUS=2
export RM_GPUS=0,1
export RM_PARALLEL=1
export STAGES=2
export RUN_HTMNPO=1
export RUN_RONPO=0
export PLAYERS=athene,armo
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
export FORCE_SCORE_ATHENE=0
export FORCE_SCORE_ARMO=0
export DECODE_BATCH_SIZE=512

mkdir -p /ext_hdd/sjkim/mnpo/logs
bash run_qwen_online_htmnpo_ronpo.sh 2>&1 | tee /ext_hdd/sjkim/mnpo/logs/htmnpo_stage2_resume_mnpo_$(date +%Y%m%d_%H%M%S).log
