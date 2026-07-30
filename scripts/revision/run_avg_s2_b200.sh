#!/usr/bin/env bash
# Stage-2 continuation of a repaired 1.5B averaged-oracle baseline on a B200
# host, via the standard online pipeline (decode -> 3-RM scoring -> avg pairs
# -> precompute -> train). Candidate-d hyperparameters throughout.
#   METHOD=sppo GPUS=0,1 bash scripts/revision/run_avg_s2_b200.sh
set -euo pipefail

METHOD="${METHOD:?set METHOD=sppo|inpo}"
GPUS="${GPUS:?set GPUS, e.g. 0,1}"
SJ=/NHNHOME/AIPR/sjkim
PROJECT_ROOT="$SJ/MNPO_rev_20260720"
RUN_ROOT="$SJ/avg_s2_20260720"
PARENT="$SJ/baseline_repair_1p5b_20260714/candidates/repair1p5b_${METHOD}_d_s42"
test -f "$PARENT/model.safetensors"

mkdir -p "$RUN_ROOT"/{out,work,logs,wandb,eval_generations}
export HF_HOME="$SJ/baseline_repair_1p5b_20260714/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export WANDB_DIR="$RUN_ROOT/wandb"
export WANDB_ENTITY=promotion-kim WANDB_PROJECT=mnpo WANDB_MODE=online
export MNPO_DISABLE_APEX=1 TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME=lo

RUN_INPO=0; RUN_SPPO=0
[[ "$METHOD" == "inpo" ]] && RUN_INPO=1
[[ "$METHOD" == "sppo" ]] && RUN_SPPO=1

USE_EXT_CACHE=0 \
CACHE_DIR="$HF_HUB_CACHE" \
PYTHON_TRAIN="$SJ/venv_clean/bin/python" \
PYTHON_INFER="$SJ/venv_clean/bin/python" \
CUDA_VISIBLE_DEVICES="$GPUS" \
RM_GPUS="$GPUS" RM_PARALLEL=1 RM_BATCH_SIZE=8 RM_SAMPLE_BATCH_SIZE=32 \
STAGES=2 RUN_HTMNPO=0 RUN_RONPO=0 RUN_INPO="$RUN_INPO" RUN_SPPO="$RUN_SPPO" \
SHARE_STAGE1_BASE_DATA=0 BASELINE_REUSE_RONPO_DATA=0 \
BASELINE_STAGE1_POLICY="$PARENT" \
WORK_ROOT="$RUN_ROOT/work/$METHOD" \
OUTPUT_ROOT="$RUN_ROOT/out" \
ACCELERATE_CONFIG="${ACC_CFG:-$PROJECT_ROOT/accelerate_configs/single_gpu.yaml}" \
NUM_PROCESSES=1 \
DECODE_ATTENTION_BACKEND= \
INPO_ETA=0.01 SPPO_ETA=0.01 \
TRAIN_LEARNING_RATE=2.5e-7 \
TRAIN_COMMON_EXTRA_ARGS="--reference_anchor_weight=0.05 --preference_sft_weight=0.005" \
TRAIN_GENERATE_DURING_EVAL=false \
TRAIN_EVAL_GENERATION_OUTPUT_DIR="$RUN_ROOT/eval_generations" \
TRAIN_EVAL_STEPS=400 TRAIN_SAVE_STEPS=400 TRAIN_SAVE_TOTAL_LIMIT=1 \
bash "$PROJECT_ROOT/run_qwen_online_htmnpo_ronpo.sh" \
  2>&1 | tee "$RUN_ROOT/logs/${METHOD}_s2_$(date +%m%d_%H%M).log"
