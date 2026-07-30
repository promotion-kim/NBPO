#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: $0 GPU:OBJECTIVE:SHARD [...]" >&2; exit 2; }
PROJECT=${PROJECT:-/NHNHOME/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/AIPR/sjkim/venv_clean}
ROOT=${ROOT:-/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1}
P7=${P7:?P7 experiment path is required}; OUT=$P7/stage3_eval
REWARD=$ROOT/flagship_20260712/cache/huggingface/models--PKU-Alignment--beaver-7b-v1.0-reward/snapshots/375cd6a9f0d7e339d2199b05ba129a4a8906596d
COST=$ROOT/flagship_20260712/cache/huggingface/models--PKU-Alignment--beaver-7b-v1.0-cost/snapshots/c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28
mkdir -p "$OUT"/{score_shards,logs}
run_one() {
  local gpu=$1 obj=$2 shard=$3 scorer model rev out
  case "$obj" in helpfulness) scorer=$PROJECT/on_policy_data_gen/rm_beaver_reward.py; model=$REWARD; rev=375cd6a9f0d7e339d2199b05ba129a4a8906596d ;; harmlessness) scorer=$PROJECT/on_policy_data_gen/rm_beaver_cost.py; model=$COST; rev=c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28 ;; *) exit 2;; esac
  out=$OUT/score_shards/${obj}_${shard}.jsonl; [[ -s "$out" ]] && return
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$PROJECT TOKENIZERS_PARALLELISM=false "$VENV/bin/python" "$scorer" --model_name "$model" --model_revision "$rev" --input_file "$OUT/shards/shard_${shard}.jsonl" --output_file "$out" --batch_size 16 --sample_batch_size 4 --max_seq_length 4096 > "$OUT/logs/${obj}_${shard}_$(hostname).log" 2>&1
}
declare -A BY_GPU=(); for item in "$@"; do IFS=: read -r gpu obj shard <<< "$item"; BY_GPU[$gpu]="${BY_GPU[$gpu]:-} $obj:$shard"; done
pids=(); for gpu in "${!BY_GPU[@]}"; do ( for item in ${BY_GPU[$gpu]}; do IFS=: read -r obj shard <<< "$item"; run_one "$gpu" "$obj" "$shard"; done ) & pids+=("$!"); done; for pid in "${pids[@]}"; do wait "$pid"; done
