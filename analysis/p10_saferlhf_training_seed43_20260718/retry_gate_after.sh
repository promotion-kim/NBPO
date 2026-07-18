#!/usr/bin/env bash
# Retry a gate only when an earlier infrastructure attempt produced no gate JSON.
set -euo pipefail
[[ $# -eq 7 ]] || { echo "usage: $0 PROJECT EXP STAGE ARM GPU AFTER_ARM LOG" >&2; exit 2; }
PROJECT=$1; EXP=$2; STAGE=$3; ARM=$4; GPU=$5; AFTER=$6; LOG=$7
ROOT=$EXP/${STAGE}_stability_p8_locked_panel/gates
mkdir -p "$(dirname "$LOG")"
{
  echo "$(date -Is) infrastructure-only gate retry for $ARM queued after $AFTER"
  while [[ ! -s "$ROOT/$AFTER.json" ]]; do sleep 20; done
  if [[ -s "$ROOT/$ARM.json" ]]; then
    echo "$(date -Is) $ARM already has a terminal gate JSON; no retry"
    exit 0
  fi
  echo "$(date -Is) $AFTER gate is terminal and $ARM has no gate JSON; retrying"
  exec "$PROJECT/analysis/p10_saferlhf_training_seed43_20260718/wait_for_continuation_gate.sh" \
    "$PROJECT" "$EXP" "$STAGE" "$ARM" "$GPU" "$LOG"
} >>"$LOG" 2>&1
