#!/usr/bin/env bash
# Score fixed retrospective diagnostic responses with the locked Beaver heads.
# Each task is GPU:OBJECTIVE:SHARD. Tasks sharing a GPU run sequentially; GPU
# workers run concurrently. This creates only JSONL scores, never new decodes.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 GPU:OBJECTIVE:SHARD [...]" >&2
  exit 2
fi

PROJECT=${PROJECT:-/NHNHOME/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/AIPR/sjkim/venv_clean}
ROOT=${ROOT:-/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1}
EXP=${EXP:-$PROJECT/results/p5_8b_robust_stage1_stage2_20260717}
OUT=$EXP/stage2/full_raw_test_conflict_retrospective_diagnostic
REWARD=$ROOT/flagship_20260712/cache/huggingface/models--PKU-Alignment--beaver-7b-v1.0-reward/snapshots/375cd6a9f0d7e339d2199b05ba129a4a8906596d
COST=$ROOT/flagship_20260712/cache/huggingface/models--PKU-Alignment--beaver-7b-v1.0-cost/snapshots/c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28
EXPECTED=693
TASKS=("$@")

mkdir -p "$OUT"/{score_shards,scores,logs}

run_one() {
  local gpu=$1 objective=$2 shard=$3 scorer model revision output count
  case "$objective" in
    helpfulness)
      scorer="$PROJECT/on_policy_data_gen/rm_beaver_reward.py"
      model=$REWARD
      revision=375cd6a9f0d7e339d2199b05ba129a4a8906596d
      ;;
    harmlessness)
      scorer="$PROJECT/on_policy_data_gen/rm_beaver_cost.py"
      model=$COST
      revision=c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28
      ;;
    *) echo "unknown objective $objective" >&2; return 2 ;;
  esac
  output="$OUT/score_shards/${objective}_${shard}.jsonl"
  if [[ -f "$output" ]]; then
    count=$(wc -l < "$output")
    if (( count > 0 )); then
      echo "skip complete ${objective}_${shard} rows=$count"
      return 0
    fi
  fi
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$PROJECT" TOKENIZERS_PARALLELISM=false \
    "$VENV/bin/python" "$scorer" --model_name "$model" --model_revision "$revision" \
    --input_file "$OUT/shards/shard_${shard}.jsonl" --output_file "$output" \
    --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 \
    > "$OUT/logs/${objective}_${shard}.log" 2>&1
}

declare -A by_gpu=()
for task in "${TASKS[@]}"; do
  IFS=: read -r gpu objective shard <<< "$task"
  [[ -n "${gpu:-}" && -n "${objective:-}" && -n "${shard:-}" ]] || { echo "bad task $task" >&2; exit 2; }
  by_gpu[$gpu]="${by_gpu[$gpu]:-} ${objective}:${shard}"
done

pids=()
for gpu in "${!by_gpu[@]}"; do
  (
    for part in ${by_gpu[$gpu]}; do
      IFS=: read -r objective shard <<< "$part"
      run_one "$gpu" "$objective" "$shard"
    done
  ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
date -Is > "$OUT/SCORE_WORKER_COMPLETE_$(hostname)"
