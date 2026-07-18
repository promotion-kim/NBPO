#!/usr/bin/env bash
# Read secrets silently from the controlling terminal; never persist or echo them.
set -euo pipefail
[[ $# -eq 7 ]] || { echo "usage: $0 PROJECT VENV ROOT BASE HOST GPUS START_UPLOADER" >&2; exit 2; }
PROJECT=$1; VENV=$2; ROOT=$3; BASE=$4; HOST=$5; GPUS=$6; START_UPLOADER=$7
IFS= read -r -s WANDB_API_KEY; printf '\n'; export WANDB_API_KEY
if [[ $START_UPLOADER == 1 ]]; then IFS= read -r -s HF_TOKEN; printf '\n'; export HF_TOKEN; fi
DIR=$ROOT/stage4/launcher; mkdir -p "$DIR"
IFS=, read -ra GPU_ARRAY <<< "$GPUS"
for gpu in "${GPU_ARRAY[@]}"; do
  pidfile=$DIR/worker_${HOST}_g${gpu}.pid
  if [[ -s $pidfile ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then continue; fi
  nohup "$VENV/bin/python" "$PROJECT/analysis/p13_saferlhf_seed44_stage4_20260718/worker.py" \
    --project "$PROJECT" --venv "$VENV" --root "$ROOT" --base "$BASE" --gpu "$gpu" \
    > "$DIR/worker_${HOST}_g${gpu}.log" 2>&1 < /dev/null & echo $! > "$pidfile"
done
nohup bash "$PROJECT/analysis/p13_saferlhf_seed44_stage4_20260718/monitor_30min.sh" "$ROOT" "$HOST" \
  > "$DIR/monitor_${HOST}.log" 2>&1 < /dev/null & echo $! > "$DIR/monitor_${HOST}.pid"
if [[ $START_UPLOADER == 1 ]]; then
  nohup "$VENV/bin/python" "$PROJECT/analysis/p13_saferlhf_seed44_stage4_20260718/upload_supervisor.py" --root "$ROOT" \
    > "$DIR/upload_supervisor.log" 2>&1 < /dev/null & echo $! > "$DIR/upload_supervisor.pid"
fi
echo "workers_launched host=$HOST gpus=$GPUS uploader=$START_UPLOADER"
