#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {sppo|inpo}" >&2
  exit 2
fi

METHOD="$1"
if [[ "$METHOD" != "sppo" && "$METHOD" != "inpo" ]]; then
  echo "unknown method: $METHOD" >&2
  exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-/home/sjkim/MNPO}"
if [[ -z "${ROOT:-}" ]]; then
  if mkdir -p /ext_hdd/sjkim/mnpo 2>/dev/null; then
    ROOT=/ext_hdd/sjkim/mnpo
  else
    ROOT=/home/sjkim/mnpo_runs/loki3
  fi
fi
WORK_ROOT="${WORK_ROOT:-$ROOT/work}"
BASE_SRC="$PROJECT_ROOT/data/qwen2.5-1.5b-instruct_online_htmnpo_ronpo/base"
JOB_CACHE="${JOB_CACHE:-$ROOT/cache/hf_${METHOD}_s1}"

mkdir -p "$ROOT"/{cache,out,logs,wandb,eval,work} "$JOB_CACHE"/{hub,datasets,transformers,torch,triton}

if [[ ! -f "$BASE_SRC/iter1/scored/train_skywork.jsonl" ]]; then
  echo "missing shared stage1 base scores under $BASE_SRC" >&2
  exit 1
fi
if [[ -L "$WORK_ROOT/base" ]]; then
  ln -sfn "$BASE_SRC" "$WORK_ROOT/base"
elif [[ -e "$WORK_ROOT/base" ]]; then
  echo "$WORK_ROOT/base exists and is not a symlink; refusing to overwrite it" >&2
  exit 1
else
  ln -s "$BASE_SRC" "$WORK_ROOT/base"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$PROJECT_ROOT"
export PYTHON_TRAIN="${PYTHON_TRAIN:-/home/sjkim/anaconda3/envs/mnpo_train/bin/python}"
export PYTHON_INFER="${PYTHON_INFER:-/home/sjkim/anaconda3/envs/mnpo_infer/bin/python}"
export PYTHON_RM_SKYWORK="$PYTHON_INFER"
export PYTHON_RM_ATHENE="$PYTHON_INFER"
export PYTHON_RM_ARMO="$PYTHON_TRAIN"

export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export STAGES=1
export RUN_HTMNPO=0
export RUN_RONPO=0
export RUN_INPO=0
export RUN_SPPO=0
if [[ "$METHOD" == "inpo" ]]; then
  export RUN_INPO=1
else
  export RUN_SPPO=1
fi

export BASELINE_REUSE_RONPO_DATA=1
export SHARE_STAGE1_BASE_DATA=1
export BASELINE_WAIT_FOR_SHARED_DATA=0
export BASELINE_COMPACT_NAMES=1
export OUTPUT_ROOT="$ROOT/out"
export WORK_ROOT
export ACCELERATE_CONFIG="$PROJECT_ROOT/accelerate_configs/single_gpu.yaml"
export TRAIN_EVAL_GENERATION_OUTPUT_DIR="$ROOT/eval"
export USE_EXT_CACHE=0

export CACHE_DIR="$JOB_CACHE/hub"
export HF_HOME="$JOB_CACHE"
export HUGGINGFACE_HUB_CACHE="$JOB_CACHE/hub"
export HF_HUB_CACHE="$JOB_CACHE/hub"
export HF_DATASETS_CACHE="$JOB_CACHE/datasets"
export TRANSFORMERS_CACHE="$JOB_CACHE/transformers"
export TORCH_HOME="$JOB_CACHE/torch"
export TRITON_CACHE_DIR="$JOB_CACHE/triton"
if mkdir -p "/tmp/sjkim_mnpo_${METHOD}_triton" 2>/dev/null; then
  export TRITON_CACHE_DIR="/tmp/sjkim_mnpo_${METHOD}_triton"
fi

export FORCE_DECODE=0
export FORCE_SCORE=0
export FORCE_BUILD_PAIRS=0
export FORCE_PRECOMPUTE=0
export TRAIN_GENERATE_DURING_EVAL=false
export TRAIN_PER_DEVICE_BATCH_SIZE="${TRAIN_PER_DEVICE_BATCH_SIZE:-2}"
export TRAIN_PER_DEVICE_EVAL_BATCH_SIZE="${TRAIN_PER_DEVICE_EVAL_BATCH_SIZE:-2}"
export TRAIN_GRADIENT_ACCUMULATION_STEPS="${TRAIN_GRADIENT_ACCUMULATION_STEPS:-8}"
export PRECOMPUTE_BATCH_SIZE="${PRECOMPUTE_BATCH_SIZE:-4}"
export PRECOMPUTE_MAX_LENGTH=2048
export PRECOMPUTE_MAX_PROMPT_LENGTH=1800
export TRAIN_MAX_LENGTH=2048
export TRAIN_MAX_PROMPT_LENGTH=1800
export NUM_PROCESSES=1
export WANDB_PROJECT="${WANDB_PROJECT:-mnpo}"
export WANDB_ENTITY="${WANDB_ENTITY:-promotion-kim}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="$ROOT/wandb"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

cd "$PROJECT_ROOT"
echo "[start] ${METHOD}_s1 on $(hostname) at $(date -Is)"
echo "[env] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES batch=$TRAIN_PER_DEVICE_BATCH_SIZE accum=$TRAIN_GRADIENT_ACCUMULATION_STEPS"
nvidia-smi || true
"$PYTHON_TRAIN" - <<'PY'
import torch, transformers, accelerate, datasets, trl, peft, wandb
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count(), "bf16", torch.cuda.is_bf16_supported())
print("transformers", transformers.__version__, "accelerate", accelerate.__version__)
print("datasets", datasets.__version__, "trl", trl.__version__, "peft", peft.__version__, "wandb", wandb.__version__)
PY

bash "$PROJECT_ROOT/run_qwen_online_htmnpo_ronpo.sh" 2>&1 | tee "$ROOT/logs/${METHOD}_s1_gpu${CUDA_VISIBLE_DEVICES}_$(date +%Y%m%d_%H%M%S).log"
