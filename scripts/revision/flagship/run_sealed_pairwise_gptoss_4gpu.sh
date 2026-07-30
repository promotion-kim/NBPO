#!/usr/bin/env bash
set -euo pipefail

PROJECT=${1:?project path required}
WORK=${2:?sealed work path required}
PYTHON=${3:?python path required}
JUDGE_MODEL_PATH=${4:?judge model snapshot path required}
PROTOCOL="$WORK/independent_judge/protocol_lock.json"
JUDGMENTS="$WORK/independent_judge/judgments"
LOGS="$WORK/independent_judge/logs"
RESULTS="$WORK/independent_judge/results"
mkdir -p "$JUDGMENTS" "$LOGS" "$RESULTS"

pids=()
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$shard" \
  TORCH_CUDNN_SDPA_ENABLED=0 \
  TOKENIZERS_PARALLELISM=false \
  WANDB_MODE=online \
  "$PYTHON" "$PROJECT/scripts/revision/flagship/judge_sealed_pairwise_gptoss.py" \
    --work "$WORK" \
    --protocol-lock "$PROTOCOL" \
    --judge-model-path "$JUDGE_MODEL_PATH" \
    --output "$JUDGMENTS/shard_${shard}.jsonl" \
    --shard-index "$shard" \
    --num-shards 4 \
    --batch-size 256 \
    >"$LOGS/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "At least one judge shard failed; aggregation is refused." >&2
  exit 4
fi

WANDB_MODE=online "$PYTHON" "$PROJECT/scripts/revision/flagship/aggregate_sealed_pairwise_gptoss.py" \
  --protocol-lock "$PROTOCOL" \
  --judgment-dir "$JUDGMENTS" \
  --output-dir "$RESULTS" \
  >"$LOGS/aggregate.log" 2>&1
