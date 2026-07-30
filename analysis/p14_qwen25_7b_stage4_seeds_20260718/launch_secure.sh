#!/usr/bin/env bash
# Read secrets from the controlling TTY and pass them only through process environments.
set -euo pipefail
[[ $# -eq 5 ]] || { echo "usage: $0 PROJECT TRAIN_PY INFER_PY ROOT CACHE" >&2; exit 2; }
PROJECT=$1; TRAIN_PY=$2; INFER_PY=$3; ROOT=$4; CACHE=$5
IFS= read -r -s WANDB_API_KEY; printf '\n'; export WANDB_API_KEY
IFS= read -r -s HF_TOKEN; printf '\n'; export HF_TOKEN
DIR=$ROOT/launcher; mkdir -p "$DIR"
while [[ ! -s $ROOT/shared/stage1_pool/PREPARED ]]; do sleep 20; done
for gpu in 0 1 2 3; do
  nohup env -u HF_TOKEN "$TRAIN_PY" "$PROJECT/analysis/p14_qwen25_7b_stage4_seeds_20260718/worker.py" \
    --project "$PROJECT" --train-python "$TRAIN_PY" --infer-python "$INFER_PY" --root "$ROOT" --cache "$CACHE" --gpu "$gpu" \
    > "$DIR/worker_g${gpu}.log" 2>&1 < /dev/null &
  pid=$!; echo "$pid" > "$DIR/worker_g${gpu}.pid"; sleep 2; kill -0 "$pid"
done
nohup env -u WANDB_API_KEY "$TRAIN_PY" "$PROJECT/analysis/p14_qwen25_7b_stage4_seeds_20260718/upload_supervisor.py" --root "$ROOT" \
  > "$DIR/upload_supervisor.log" 2>&1 < /dev/null &
pid=$!; echo "$pid" > "$DIR/upload_supervisor.pid"; sleep 2; kill -0 "$pid"
nohup env -u WANDB_API_KEY -u HF_TOKEN bash "$PROJECT/analysis/p14_qwen25_7b_stage4_seeds_20260718/evaluation_supervisor.sh" "$PROJECT" "$TRAIN_PY" "$ROOT" "$CACHE" \
  > "$DIR/evaluation_supervisor.log" 2>&1 < /dev/null &
pid=$!; echo "$pid" > "$DIR/evaluation_supervisor.pid"; sleep 2; kill -0 "$pid"
nohup env -u WANDB_API_KEY -u HF_TOKEN bash "$PROJECT/analysis/p14_qwen25_7b_stage4_seeds_20260718/monitor_30min.sh" "$ROOT" \
  > "$DIR/monitor.log" 2>&1 < /dev/null &
pid=$!; echo "$pid" > "$DIR/monitor.pid"; sleep 2; kill -0 "$pid"
echo "launched workers=4 uploader=1 evaluator=1 monitor=1"
