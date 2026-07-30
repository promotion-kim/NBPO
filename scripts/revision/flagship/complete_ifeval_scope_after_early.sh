#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:?set ROOT}"
PROJ="${PROJ:?set PROJ}"
EARLY_PID="${EARLY_PID:?set EARLY_PID}"
OLD_IFEVAL_PID="${OLD_IFEVAL_PID:?set OLD_IFEVAL_PID}"
EVALPY="$ROOT/venv_lm_eval/bin/python"

alive_non_zombie() {
  local pid="$1" state
  kill -0 "$pid" 2>/dev/null || return 1
  state="$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || true)"
  [[ "$state" != "Z" ]]
}

while alive_non_zombie "$EARLY_PID"; do sleep 20; done
while alive_non_zombie "$OLD_IFEVAL_PID"; do sleep 20; done

# The old in-memory runner predates the explicit k-only scope amendment. Its
# measured JSONs remain intact; this idempotent pass fills only missing methods.
tmux kill-session -t ronpo_ifeval 2>/dev/null || true
tmux new-session -d -s ronpo_ifeval \
  "cd $PROJ && PYTHONPATH=$PROJ $EVALPY scripts/revision/flagship/run_seed42_ifeval.py --root $ROOT --python $EVALPY --work $ROOT/eval/p2_ifeval_seed42 --base-revision b968826d9c46dd6066d109eabc6255188de91218 --stop-at 2026-07-15T18:00:00+09:00 > $ROOT/eval/p2_ifeval_seed42/logs/runner_scope_20260714.log 2>&1"

while tmux has-session -t ronpo_ifeval 2>/dev/null; do sleep 20; done
mkdir -p "$ROOT/eval/p2_ifeval_seed42/results"
"$EVALPY" "$PROJ/scripts/revision/flagship/aggregate_seed42_ifeval.py" \
  --work "$ROOT/eval/p2_ifeval_seed42" \
  --output-dir "$ROOT/eval/p2_ifeval_seed42/results" \
  >> "$ROOT/eval/p2_ifeval_seed42/logs/aggregate_scope_20260714.log" 2>&1
