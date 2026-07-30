#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 CANDIDATE_NAME GPU_INDEX" >&2
  exit 2
fi

NAME="$1"
GPU="$2"
[[ "$GPU" =~ ^[0-3]$ ]] || { echo "GPU must be one of the four authorized indices 0-3" >&2; exit 2; }

PROJECT_ROOT="${PROJECT_ROOT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/MNPO_rev_20260710}"
RUN_ROOT="${RUN_ROOT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/baseline_repair_1p5b_20260714}"
PYTHON="${PYTHON_INFER:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/revision_qwen3_8b/full_iter1/flagship_20260712/venv_lm_eval/bin/python}"
MODEL="$RUN_ROOT/candidates/$NAME"
OUT="$RUN_ROOT/eval/generations/$NAME"
LOG="$RUN_ROOT/logs/decode_${NAME}.log"

[[ -f "$MODEL/model.safetensors" ]] || { echo "missing consolidated model: $MODEL/model.safetensors" >&2; exit 3; }
mkdir -p "$OUT" "$RUN_ROOT/eval/metadata"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$PROJECT_ROOT"
export HF_HOME="$RUN_ROOT/cache/huggingface_eval"
export TOKENIZERS_PARALLELISM=false
# vLLM 0.24 defaults its V1 engine core to a spawned subprocess.  The
# repository's legacy module entry point predates Python's spawn-safe main
# guard, so use vLLM's supported one-process executor for this single-GPU
# decode.  This changes process topology only, not sampling semantics.
export VLLM_ENABLE_V1_MULTIPROCESSING=0

COMMAND=(
  "$PYTHON" -u -m on_policy_data_gen.decode
  --data_dir "$PROJECT_ROOT/data/gemma2_ufb_part1_test.jsonl"
  --model "$MODEL"
  --seeds 42
  --output_dir "$OUT"
  --num_gpu 1
  --temperature 0.7
  --top_p 0.9
  --max_tokens 2048
  --batch_size 647
  --attention_backend XFORMERS
  --dtype bfloat16
  --gpu_memory_utilization 0.6
)

printf '%q ' "${COMMAND[@]}" > "$RUN_ROOT/eval/metadata/${NAME}_decode_command.txt"
printf '\n' >> "$RUN_ROOT/eval/metadata/${NAME}_decode_command.txt"
{
  echo "started_at=$(date -Is)"
  echo "physical_gpu=$GPU"
  echo "model=$MODEL"
  echo "seed=42"
  echo "temperature=0.7"
  echo "top_p=0.9"
  echo "max_new_tokens=2048"
  echo "dtype=bfloat16"
  echo "enable_thinking=false"
  echo "attention_backend=XFORMERS"
  echo "vllm_enable_v1_multiprocessing=0"
} > "$RUN_ROOT/eval/metadata/${NAME}_decode.env"

"${COMMAND[@]}" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
{
  echo "finished_at=$(date -Is)"
  echo "exit_code=$status"
} >> "$RUN_ROOT/eval/metadata/${NAME}_decode.env"
exit "$status"
