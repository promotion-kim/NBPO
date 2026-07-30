#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --method {sppo|inpo} --candidate ID --gpu N[,N] --eta X --reference-anchor X --sft-anchor X --learning-rate X [--max-steps N]" >&2
  exit 2
}

METHOD=""
CANDIDATE=""
GPU=""
ETA=""
REFERENCE_ANCHOR=""
SFT_ANCHOR=""
LEARNING_RATE=""
MAX_STEPS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) METHOD="$2"; shift 2 ;;
    --candidate) CANDIDATE="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --eta) ETA="$2"; shift 2 ;;
    --reference-anchor) REFERENCE_ANCHOR="$2"; shift 2 ;;
    --sft-anchor) SFT_ANCHOR="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    *) usage ;;
  esac
done

[[ "$METHOD" == "sppo" || "$METHOD" == "inpo" ]] || usage
[[ -n "$CANDIDATE" && -n "$GPU" && -n "$ETA" && -n "$REFERENCE_ANCHOR" && -n "$SFT_ANCHOR" && -n "$LEARNING_RATE" ]] || usage
[[ "$GPU" =~ ^[0-3](,[0-3])?$ ]] || { echo "GPU list must contain one or two of the authorized indices 0,1,2,3" >&2; exit 2; }
IFS=',' read -ra GPU_LIST <<< "$GPU"
[[ "${GPU_LIST[0]}" != "${GPU_LIST[-1]}" || "${#GPU_LIST[@]}" -eq 1 ]] || { echo "GPU list contains a duplicate" >&2; exit 2; }
NUM_PROCESSES="${#GPU_LIST[@]}"
if [[ "$NUM_PROCESSES" -eq 2 ]]; then
  MICRO_BATCH=1
else
  MICRO_BATCH=2
fi
MAIN_PROCESS_PORT="$((29600 + GPU_LIST[0]))"

PROJECT_ROOT="${PROJECT_ROOT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/MNPO_rev_20260710}"
RUN_ROOT="${RUN_ROOT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/baseline_repair_1p5b_20260714}"
PYTHON="${PYTHON:-$RUN_ROOT/venv_train/bin/python}"
PRECOMPUTED="${PRECOMPUTED:-$RUN_ROOT/data/avg_oracle_precomputed}"
CONFIG="$PROJECT_ROOT/training_configs/$METHOD/qwen2.5-1.5b-instruct-${METHOD}-avg-multiobj-iter1.yaml"

SUFFIX=""
if [[ -n "$MAX_STEPS" ]]; then
  SUFFIX="_smoke${MAX_STEPS}"
fi
RUN_NAME="repair1p5b_${METHOD}_${CANDIDATE}_s42${SUFFIX}"
OUTPUT_DIR="$RUN_ROOT/candidates/$RUN_NAME"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$RUN_ROOT/cache/huggingface" "$RUN_ROOT/wandb"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$PROJECT_ROOT"
export HF_HOME="$RUN_ROOT/cache/huggingface"
export HF_HUB_CACHE="$RUN_ROOT/cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="$RUN_ROOT/cache/huggingface/hub"
export TRANSFORMERS_CACHE="$RUN_ROOT/cache/huggingface/transformers"
export WANDB_ENTITY="promotion-kim"
export WANDB_PROJECT="mnpo"
export WANDB_MODE="online"
export WANDB_DIR="$RUN_ROOT/wandb"
export WANDB_RUN_ID="$RUN_NAME"
export WANDB_RESUME="allow"
export TOKENIZERS_PARALLELISM="false"
export MNPO_DISABLE_APEX="1"

MAX_STEPS_ARGS=()
if [[ -n "$MAX_STEPS" ]]; then
  MAX_STEPS_ARGS=(--max_steps="$MAX_STEPS" --do_eval=false --evaluation_strategy=no --save_strategy=no)
fi

COMMAND=(
  "$PYTHON" -m accelerate.commands.launch
  --config_file "$PROJECT_ROOT/accelerate_configs/deepspeed_zero3_single_process.yaml"
  --num_processes "$NUM_PROCESSES"
  --main_process_port "$MAIN_PROCESS_PORT"
  -m mnpo_scripts.run_mnpo
  "$CONFIG"
  --model_name_or_path=Qwen/Qwen2.5-1.5B-Instruct
  --dataset_mixer="$PRECOMPUTED:1.0"
  --output_dir="$OUTPUT_DIR"
  --run_name="$RUN_NAME"
  --loss_type="$METHOD"
  --eta="$ETA"
  --ratio=0.3333
  --max_history_t=1
  --history_weights=1.0
  --reference_anchor_weight="$REFERENCE_ANCHOR"
  --preference_sft_weight="$SFT_ANCHOR"
  --learning_rate="$LEARNING_RATE"
  --lr_scheduler_type=cosine
  --warmup_ratio=0.1
  --optim=adamw_torch
  --weight_decay=0.0
  --num_train_epochs=1
  --seed=42
  --bf16=true
  --per_device_train_batch_size="$MICRO_BATCH"
  --per_device_eval_batch_size="$MICRO_BATCH"
  --gradient_accumulation_steps=8
  --gradient_checkpointing=true
  --max_length=2048
  --max_prompt_length=1800
  --generate_during_eval=false
  --evaluation_strategy=steps
  --eval_steps=400
  --save_strategy=steps
  --save_steps=400
  --save_total_limit=1
  --logging_steps=5
  --report_to=wandb
  "${MAX_STEPS_ARGS[@]}"
)

printf '%q ' "${COMMAND[@]}" > "$OUTPUT_DIR/exact_command.txt"
printf '\n' >> "$OUTPUT_DIR/exact_command.txt"
{
  echo "started_at=$(date -Is)"
  echo "hostname=$(hostname)"
  echo "physical_gpu=$GPU"
  echo "num_processes=$NUM_PROCESSES"
  echo "per_device_train_batch_size=$MICRO_BATCH"
  echo "effective_batch_size=$((MICRO_BATCH * 8 * NUM_PROCESSES))"
  echo "wandb_run_id=$WANDB_RUN_ID"
  echo "method=$METHOD"
  echo "candidate=$CANDIDATE"
  echo "eta=$ETA"
  echo "reference_anchor_weight=$REFERENCE_ANCHOR"
  echo "preference_sft_weight=$SFT_ANCHOR"
  echo "learning_rate=$LEARNING_RATE"
  echo "max_steps=${MAX_STEPS:-1184}"
} > "$OUTPUT_DIR/run_metadata.env"

echo "[$(date -Is)] launching $RUN_NAME on physical GPU $GPU"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
"${COMMAND[@]}" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"
status=${PIPESTATUS[0]}
echo "finished_at=$(date -Is)" >> "$OUTPUT_DIR/run_metadata.env"
echo "exit_code=$status" >> "$OUTPUT_DIR/run_metadata.env"
exit "$status"
