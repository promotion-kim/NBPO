#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

WORK_DIR="${WORK_DIR:-/ext_hdd/sjkim/mnpo/eval/ronpo_stage2_ifeval_20260625}"
LOG_DIR="${WORK_DIR}/logs"
DATASETS="${DATASETS:-ifeval}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-20}"
EVAL_PYTHON="${EVAL_PYTHON:-/home/sjkim/anaconda3/envs/evalscope/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/home/sjkim/anaconda3/envs/mnpo_infer/bin/python}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:---gpu-memory-utilization 0.9 --dtype bfloat16}"

mkdir -p "$LOG_DIR"

NAMES=(
  "baseline"
  "ronpo_s2_final"
  "ronpo_s2_ckpt1400"
  "htmnpo_skywork_s2"
  "htmnpo_athene_s2"
  "htmnpo_armo_s2"
)

MODELS=(
  "Qwen/Qwen2.5-1.5B-Instruct"
  "/ext_hdd/sjkim/mnpo/outputs_ronpo_h200/qwen2.5-1.5b-instruct_ronpo_stage2_relative_lr2e8_od2g2/checkpoint-2457"
  "/ext_hdd/sjkim/mnpo/outputs_ronpo_h200/qwen2.5-1.5b-instruct_ronpo_stage2_relative_lr2e8_od2g2/checkpoint-1400"
  "/ext_hdd/sjkim/mnpo/ht_stage1_out/qwen2.5-1.5b-instruct_htmnpo_skywork_online_multiobj_stage_2"
  "/ext_hdd/sjkim/mnpo/ht_stage1_out/qwen2.5-1.5b-instruct_htmnpo_athene_online_multiobj_stage_2"
  "/ext_hdd/sjkim/mnpo/ht_stage1_out/qwen2.5-1.5b-instruct_htmnpo_armo_online_multiobj_stage_2"
)

PAIR_GPUS=(1 2)
PAIR_PORTS=(9101 9102)

echo "[start] Stage-2 IFEval suite at $(date -Is)"
echo "[work] ${WORK_DIR}"
echo "[datasets] ${DATASETS}"

run_one() {
  local idx="$1"
  local slot="$2"
  local name="${NAMES[$idx]}"
  local model="${MODELS[$idx]}"
  local gpu="${PAIR_GPUS[$slot]}"
  local port="${PAIR_PORTS[$slot]}"

  echo "[ifeval:${name}] gpu=${gpu} port=${port} model=${model}"
  MODEL_NAME="$model" \
  MODEL_BASENAME="$name" \
  GPU_IDS="$gpu" \
  PORT="$port" \
  TENSOR_PARALLEL_SIZE=1 \
  LOG_DIR="$LOG_DIR" \
  EVAL_PYTHON="$EVAL_PYTHON" \
  VLLM_PYTHON="$VLLM_PYTHON" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  EXTRA_VLLM_ARGS="$EXTRA_VLLM_ARGS" \
  TASK_ARGS="--datasets ${DATASETS} --eval-batch-size ${EVAL_BATCH_SIZE}" \
  bash "$PROJECT_ROOT/evalscope/run_vllm_eval.sh"
}

for ((i = 0; i < ${#MODELS[@]}; i += 2)); do
  pids=()
  for slot in 0 1; do
    idx=$((i + slot))
    if ((idx >= ${#MODELS[@]})); then
      continue
    fi
    run_one "$idx" "$slot" &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done

echo "[done] Stage-2 IFEval suite at $(date -Is)"
