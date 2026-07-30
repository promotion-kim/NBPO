#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 4 ]]; then
  echo "usage: $0 {skywork|athene|armo} GPU_INDEX [NUM_SHARDS] [SHARD_INDEX]" >&2
  exit 2
fi

OBJECTIVE="$1"
GPU="$2"
NUM_SHARDS="${3:-1}"
SHARD_INDEX="${4:-0}"
[[ "$OBJECTIVE" =~ ^(skywork|athene|armo)$ ]] || exit 2
[[ "$GPU" =~ ^[0-3]$ ]] || exit 2

PROJECT_ROOT="${PROJECT_ROOT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/MNPO_rev_20260710}"
RUN_ROOT="${RUN_ROOT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/baseline_repair_1p5b_20260714}"
PYTHON="${PYTHON_RM:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/revision_qwen3_8b/full_iter1/flagship_20260712/venv_lm_eval/bin/python}"
INPUT="$RUN_ROOT/eval/merged_model_generations.json"
CACHE="$RUN_ROOT/cache/reward_models"
SUFFIX=""
if [[ "$NUM_SHARDS" -gt 1 ]]; then
  SUFFIX=".shard${SHARD_INDEX}-of-${NUM_SHARDS}"
fi
OUTPUT="$RUN_ROOT/eval/scored/eval_${OBJECTIVE}${SUFFIX}.jsonl"
LOG="$RUN_ROOT/logs/rm_${OBJECTIVE}${SUFFIX}.log"

[[ -f "$INPUT" ]] || { echo "missing merged input: $INPUT" >&2; exit 3; }
mkdir -p "$RUN_ROOT/eval/scored" "$RUN_ROOT/eval/metadata"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$PROJECT_ROOT"
export HF_HOME="$CACHE"
export TOKENIZERS_PARALLELISM=false

COMMON=(--input_file "$INPUT" --output_file "$OUTPUT" --cache_dir "$CACHE" --batch_size 8 --sample_batch_size 32 --num_shards "$NUM_SHARDS" --shard_index "$SHARD_INDEX")
case "$OBJECTIVE" in
  skywork)
    COMMAND=("$PYTHON" -u -m on_policy_data_gen.rm_skywork "${COMMON[@]}" --max_seq_length 4096 --attn_implementation eager)
    ;;
  athene)
    COMMAND=("$PYTHON" -u -m on_policy_data_gen.rm_athene "${COMMON[@]}")
    ;;
  armo)
    COMMAND=("$PYTHON" -u -m on_policy_data_gen.rm_armo "${COMMON[@]}" --max_seq_length 4096 --local_files_only true)
    ;;
esac

printf '%q ' "${COMMAND[@]}" > "$RUN_ROOT/eval/metadata/rm_${OBJECTIVE}${SUFFIX}_command.txt"
printf '\n' >> "$RUN_ROOT/eval/metadata/rm_${OBJECTIVE}${SUFFIX}_command.txt"
{
  echo "started_at=$(date -Is)"
  echo "physical_gpu=$GPU"
  echo "objective=$OBJECTIVE"
  echo "num_shards=$NUM_SHARDS"
  echo "shard_index=$SHARD_INDEX"
  echo "batch_size=8"
  echo "sample_batch_size=32"
  echo "max_seq_length=4096"
} > "$RUN_ROOT/eval/metadata/rm_${OBJECTIVE}${SUFFIX}.env"

"${COMMAND[@]}" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
{
  echo "finished_at=$(date -Is)"
  echo "exit_code=$status"
} >> "$RUN_ROOT/eval/metadata/rm_${OBJECTIVE}${SUFFIX}.env"
exit "$status"
