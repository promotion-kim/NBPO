#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:?set ROOT}"
PROJ="${PROJ:?set PROJ}"
EVALPY="$ROOT/venv_lm_eval/bin/python"
EARLY_PID="${EARLY_PID:?set EARLY_PID}"
IFEVAL_PID="${IFEVAL_PID:?set IFEVAL_PID}"
active_eval_pids="${ACTIVE_EVAL_PIDS:?set ACTIVE_EVAL_PIDS}"
PROTOCOL="${PROTOCOL:-$PROJ/results/p2_scope_amendment_20260714.json}"

cmdline="$(tr '\0' ' ' < "/proc/$EARLY_PID/cmdline")"
[[ "$cmdline" == *run_seed42_academic_early.py* ]]
kill -STOP "$EARLY_PID"

for pid in ${active_eval_pids//,/ }; do
  while kill -0 "$pid" 2>/dev/null; do
    # A stopped parent cannot reap an exited worker. Treat a zombie worker as
    # finished so the recovery supervisor does not strand an idle GPU.
    state="$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || true)"
    [[ "$state" == "Z" ]] && break
    sleep 15
  done
done

tmux kill-session -t ronpo_p2_early 2>/dev/null || true
tmux kill-session -t ronpo_p2_academic 2>/dev/null || true

tmux new-session -d -s ronpo_p2_academic \
  "cd $PROJ && PYTHONPATH=$PROJ $EVALPY scripts/revision/flagship/run_seed42_academic_suite.py --root $ROOT --python $EVALPY --work $ROOT/eval/p2_academic_seed42 --ifeval-work $ROOT/eval/p2_ifeval_seed42 --protocol $PROTOCOL --project $PROJ --base-revision b968826d9c46dd6066d109eabc6255188de91218 --stop-at 2026-07-15T18:00:00+09:00 > $ROOT/eval/p2_academic_seed42/logs/runner.log 2>&1"

tmux new-session -d -s ronpo_p2_early \
  "cd $PROJ && PYTHONPATH=$PROJ/scripts/revision/flagship:$PROJ $EVALPY scripts/revision/flagship/run_seed42_academic_early.py --root $ROOT --python $EVALPY --work $ROOT/eval/p2_academic_seed42 --project $PROJ --base-revision b968826d9c46dd6066d109eabc6255188de91218 --stop-at 2026-07-15T18:00:00+09:00 --resume-ifeval-pid $IFEVAL_PID > $ROOT/eval/p2_academic_seed42/logs/early_runner_repaired.log 2>&1"
