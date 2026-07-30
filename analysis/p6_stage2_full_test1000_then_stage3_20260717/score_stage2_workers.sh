#!/usr/bin/env bash
# Score locked P6 eligible generations. Tasks on each GPU run sequentially.
set -euo pipefail
if [[ $# -lt 1 ]]; then echo "usage: $0 GPU:OBJECTIVE:SHARD [...]" >&2; exit 2; fi
PROJECT=${PROJECT:-/NHNHOME/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/AIPR/sjkim/venv_clean}
ROOT=${ROOT:-/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1}
P6=${P6:-$PROJECT/results/p6_stage2_full_test1000_then_stage3_20260717}
OUT=$P6/stage2_eval
REWARD=$ROOT/flagship_20260712/cache/huggingface/models--PKU-Alignment--beaver-7b-v1.0-reward/snapshots/375cd6a9f0d7e339d2199b05ba129a4a8906596d
COST=$ROOT/flagship_20260712/cache/huggingface/models--PKU-Alignment--beaver-7b-v1.0-cost/snapshots/c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28
mkdir -p "$OUT"/{score_shards,logs}
run_one() {
  local gpu=$1 objective=$2 shard=$3 scorer model revision out
  case "$objective" in
    helpfulness) scorer=$PROJECT/on_policy_data_gen/rm_beaver_reward.py; model=$REWARD; revision=375cd6a9f0d7e339d2199b05ba129a4a8906596d ;;
    harmlessness) scorer=$PROJECT/on_policy_data_gen/rm_beaver_cost.py; model=$COST; revision=c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28 ;;
    *) exit 2 ;;
  esac
  out=$OUT/score_shards/${objective}_${shard}.jsonl
  if [[ -s "$out" ]]; then return; fi
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$PROJECT TOKENIZERS_PARALLELISM=false "$VENV/bin/python" "$scorer" \
    --model_name "$model" --model_revision "$revision" --input_file "$OUT/shards/shard_${shard}.jsonl" --output_file "$out" \
    --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 > "$OUT/logs/${objective}_${shard}_$(hostname).log" 2>&1
}
declare -A BY_GPU=()
for task in "$@"; do IFS=: read -r gpu objective shard <<< "$task"; BY_GPU[$gpu]="${BY_GPU[$gpu]:-} $objective:$shard"; done
pids=()
for gpu in "${!BY_GPU[@]}"; do ( for task in ${BY_GPU[$gpu]}; do IFS=: read -r objective shard <<< "$task"; run_one "$gpu" "$objective" "$shard"; done ) & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid"; done
