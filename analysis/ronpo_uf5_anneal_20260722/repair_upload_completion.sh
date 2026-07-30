#!/usr/bin/env bash
set -euo pipefail
ARM=${1:?arm}
SJ=/NHNHOME/AIPR/sjkim; P=$SJ/MNPO_rev_20260720; ROOT=$SJ/ronpo_uf5_anneal_20260722
PY=$SJ/venv_clean/bin/python
while [[ ! -s "$ROOT/$ARM/stage4/STAGE_COMPLETE" ]]; do
  find "$ROOT/$ARM" -name GATE_FAILED -type f -print -quit 2>/dev/null | grep -q . && exit 0
  sleep 30
done
# STAGE_COMPLETE is written only after training and the fail-closed gate finish.
# Do not grep the tmux server command line here: the long-lived tmux server may
# retain the original run_arm.sh text after the arm process itself has exited.
[[ -s "$ROOT/$ARM/ARM_COMPLETE" ]] && exit 0
while pgrep -f "upload_stage.py .*--arm $ARM" >/dev/null; do sleep 30; done
for stage in 1 2 3 4; do
  work=$ROOT/$ARM/stage$stage; audit=$ROOT/hf_uploads/${ARM}_stage${stage}.json
  [[ -s "$work/STAGE_COMPLETE" ]]
  if ! [[ -s $audit ]] || ! "$PY" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("verified") else 1)' "$audit"; then
    "$PY" "$P/analysis/ronpo_uf5_anneal_20260722/upload_stage.py" \
      --model "$work/train" --arm "$ARM" --stage "$stage" --audit "$audit" \
      > "$work/logs/hf_upload_repair.log" 2>&1
  fi
done
for stage in 1 2 3 4; do
  work=$ROOT/$ARM/stage$stage
  find "$work/train" -type f \( -name optimizer.pt -o -name scheduler.pt -o -name rng_state.pth \) -delete
  find "$work/train" -maxdepth 1 -type d -name 'checkpoint-*' -prune -exec rm -rf {} +
done
printf '%s: %s upload-completion repair verified all four public HF audits after the earlier metadata failure and restored the normal ARM_COMPLETE terminal marker; no model, gate, or reward result was retried.\n' \
  "$(date -Is)" "$ARM" >> "$ROOT/fix_log.md"
date -Is > "$ROOT/$ARM/ARM_COMPLETE"
