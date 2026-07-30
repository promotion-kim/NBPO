#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${USE_EXT_CACHE:-1}" == "1" ]]; then
  # shellcheck source=../scripts/setup_ext_cache.sh
  source "$PROJECT_ROOT/scripts/setup_ext_cache.sh"
fi
MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME to a Hugging Face model id or local checkpoint path.}"
MODEL_BASENAME="${MODEL_BASENAME:-$(basename "$MODEL_NAME")}"
PORT="${PORT:-9000}"
GPU_IDS="${GPU_IDS:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
EVAL_PYTHON="${EVAL_PYTHON:-/home/sjkim/anaconda3/envs/evalscope/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/home/sjkim/anaconda3/envs/mnpo_infer/bin/python}"
EVAL_PYTHONPATH="${EVAL_PYTHONPATH:-$PROJECT_ROOT/.evalscope_pkgs}"
VLLM_PYTHONPATH="${VLLM_PYTHONPATH:-$PROJECT_ROOT/vllm_compat:$PROJECT_ROOT/.vllm051_pkgs}"
OUTLINES_CACHE_DIR="${OUTLINES_CACHE_DIR:-$PROJECT_ROOT/.cache/outlines}"
TASK_SCRIPT="${TASK_SCRIPT:-$PROJECT_ROOT/evalscope/run_rule_based_task.py}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/evalscope/logs}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

mkdir -p "$LOG_DIR"

RUN_LOG="$LOG_DIR/judge-${MODEL_BASENAME}.out"
VLLM_LOG="$LOG_DIR/vllm-${MODEL_BASENAME}-${PORT}.log"

echo "$(date '+%Y-%m-%d %H:%M:%S') Starting vLLM for ${MODEL_NAME}" | tee -a "$RUN_LOG"

mkdir -p "$OUTLINES_CACHE_DIR"

CUDA_VISIBLE_DEVICES="$GPU_IDS" \
PYTHONPATH="$VLLM_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
OUTLINES_CACHE_DIR="$OUTLINES_CACHE_DIR" \
"$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --served-model-name "$MODEL_BASENAME" \
  --trust-remote-code \
  --port "$PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  ${EXTRA_VLLM_ARGS:-} \
  > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}" >> "$RUN_LOG"

cleanup() {
  if kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') Stopping vLLM PID=${VLLM_PID}" >> "$RUN_LOG"
    kill "$VLLM_PID" || true
    wait "$VLLM_PID" || true
  fi
}
trap cleanup EXIT

echo "$(date '+%Y-%m-%d %H:%M:%S') Waiting for vLLM startup: ${VLLM_LOG}" >> "$RUN_LOG"
for _ in $(seq 1 120); do
  if [[ -f "$VLLM_LOG" ]] && grep -q "Application startup complete." "$VLLM_LOG"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') vLLM startup confirmed." >> "$RUN_LOG"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited before startup. Last log lines:" >> "$RUN_LOG"
    tail -n 80 "$VLLM_LOG" >> "$RUN_LOG" || true
    exit 1
  fi
  sleep 5
done

if ! grep -q "Application startup complete." "$VLLM_LOG"; then
  echo "Timed out waiting for vLLM startup. Last log lines:" >> "$RUN_LOG"
  tail -n 80 "$VLLM_LOG" >> "$RUN_LOG" || true
  exit 1
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') Starting EvalScope task ${TASK_SCRIPT}" >> "$RUN_LOG"
PYTHONPATH="$EVAL_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
"$EVAL_PYTHON" "$TASK_SCRIPT" \
  --model-name "$MODEL_BASENAME" \
  --port "$PORT" \
  ${TASK_ARGS:-} \
  >> "$RUN_LOG" 2>&1

echo "$(date '+%Y-%m-%d %H:%M:%S') Eval complete for ${MODEL_BASENAME}" >> "$RUN_LOG"
