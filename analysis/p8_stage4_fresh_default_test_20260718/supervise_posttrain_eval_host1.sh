#!/usr/bin/env bash
# Resume-safe P8 post-training evaluator. It is intentionally reward-blind until
# every frozen Stage-4 full run has written a finite completed status.
set -euo pipefail

PROJECT=${PROJECT:-/NHNHOME/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/AIPR/sjkim/venv_clean}
E=${E:-$PROJECT/results/p8_stage4_fresh_default_test_20260718}
SCRIPT_DIR=$PROJECT/analysis/p8_stage4_fresh_default_test_20260718
MANIFEST=$E/dataset_manifest/fresh_default_test_1000.jsonl
LOG=$E/logs/stage4/posttrain_supervisor.log
MODELS=(base ronpo_os_stage4 ronpo_topmass_stage4 inpo_avg_stage4 sppo_avg_stage4 simpo_stage4 ipo_stage4 dpo_stage4 ht_mnpo_harmless_stage4 ht_mnpo_helpfulness_stage4)
ARMS=(ronpo_os_stage4 ronpo_topmass_stage4 inpo_avg_stage4 sppo_avg_stage4 simpo_stage4 ipo_stage4 dpo_stage4 ht_mnpo_harmless_stage4 ht_mnpo_helpfulness_stage4)

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "$(date -Is) supervisor started; no reward scores consulted"

while :; do
  ok=1
  for arm in "${ARMS[@]}"; do
    f=$E/stage4/$arm/train/full/job_status.json
    if [[ ! -s $f ]] || ! "$VENV/bin/python" - "$f" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
raise SystemExit(0 if d.get('status') == 'completed' and d.get('returncode') == 0 and d.get('finite_metrics') else 1)
PY
    then ok=0; break; fi
  done
  [[ $ok == 1 ]] && break
  sleep 60
done
echo "$(date -Is) all full runs finite-completed; beginning frozen evaluation"
# P9's preregistered seed replication shares this host but is not part of P8.
# Do not overlap vLLM/RM evaluation with any existing compute process.
while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; do
  echo "$(date -Is) waiting for existing local compute before P8 evaluation"
  sleep 60
done
for i in 1 2 3; do
  date -Is
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
  sleep 10
done

# Base must exist before candidate gates can compare length/repetition against it.
if [[ ! -s $E/stage4_eval/gates/base.json ]]; then
  P8=$E MANIFEST=$MANIFEST bash "$SCRIPT_DIR/decode_and_gate_stage3_model.sh" base 0
fi

pending=()
for m in "${ARMS[@]}"; do [[ -s $E/stage4_eval/gates/$m.json ]] || pending+=("$m"); done
while ((${#pending[@]})); do
  pids=()
  for g in 0 1 2 3; do
    ((${#pending[@]})) || break
    m=${pending[0]}; pending=("${pending[@]:1}")
    (P8=$E MANIFEST=$MANIFEST bash "$SCRIPT_DIR/decode_and_gate_stage3_model.sh" "$m" "$g") & pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p"; done
done

MERGE=$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/merge_eval_pool.py
SHARD=$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py
CSV=$(IFS=,; echo "${MODELS[*]}")
"$VENV/bin/python" "$MERGE" --generation-root "$E/stage4_eval/generations" --models "$CSV" --seed 42 --expected-records 1000 --gate-root "$E/stage4_eval/gates" --output "$E/stage4_eval/response_pool.jsonl" --audit "$E/stage4_eval/pool_audit.json"
"$VENV/bin/python" "$SHARD" split --input "$E/stage4_eval/response_pool.jsonl" --output-dir "$E/stage4_eval/shards" --num-shards 6 --expected-records 1000

waves=(
  '0:helpfulness:0 1:helpfulness:1 2:helpfulness:2 3:helpfulness:3'
  '0:harmlessness:0 1:harmlessness:1 2:harmlessness:2 3:harmlessness:3'
  # Score the final two shards on the four physical host GPUs.  The first
  # field is a GPU index, not a shard index.
  '0:helpfulness:4 1:helpfulness:5 2:harmlessness:4 3:harmlessness:5'
)
for wave in "${waves[@]}"; do
  pids=()
  for job in $wave; do
    (P8=$E bash "$SCRIPT_DIR/score_stage3_workers.sh" "$job") & pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p"; done
done
"$VENV/bin/python" "$SCRIPT_DIR/finalize_stage4_eval.py" --project "$PROJECT" --output "$E/stage4_eval" --lock "$E/run_lock.json"
echo "$(date -Is) supervisor completed"
