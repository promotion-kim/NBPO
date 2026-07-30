#!/usr/bin/env bash
# Recover a clean current-run status only after a legacy append-log GPU job exits.
set -euo pipefail
[[ $# -eq 4 ]] || { echo "usage: $0 PROJECT EXP ARM GPU" >&2; exit 2; }
PROJECT=$1; EXP=$2; ARM=$3; GPU=$4
LOG=$EXP/logs/stage1/recover_${ARM}_gpu${GPU}.log
mkdir -p "$(dirname "$LOG")"
{
  echo "$(date -Is) status-recovery queue registered for $ARM"
  while :; do
    idle=1
    for _ in 1 2 3; do
      pids=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
      printf '%s gpu=%s compute_pids=%s\n' "$(date -Is)" "$GPU" "${pids:-none}"
      [[ -z ${pids//[[:space:]]/} ]] || { idle=0; break; }
      sleep 3
    done
    [[ $idle -eq 1 ]] && break
    sleep 20
  done
  "$PROJECT/../venv_clean/bin/python" "$PROJECT/analysis/p10_saferlhf_training_seed43_20260718/recover_legacy_full_status.py" --experiment "$EXP" --arm "$ARM"
} >>"$LOG" 2>&1
