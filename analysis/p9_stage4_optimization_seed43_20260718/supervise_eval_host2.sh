#!/usr/bin/env bash
# Resume-safe P9 evaluation.  It starts only after P8's primary report is
# complete, keeps a separate normalization pool, and cannot alter P8 results.
set -euo pipefail
PROJECT=${PROJECT:-/NHNHOME/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/AIPR/sjkim/venv_clean}
E=${E:-$PROJECT/results/p8_stage4_fresh_default_test_20260718}
SCRIPT_DIR=$PROJECT/analysis/p9_stage4_optimization_seed43_20260718
OUT=$E/p9_seed43_eval
LOG=$OUT/supervisor.log
MANDATORY=(ronpo_os_stage4_s43 ipo_stage4_s43)
OPTIONAL=ronpo_topmass_stage4_s43
mkdir -p "$OUT"
exec >>"$LOG" 2>&1
echo "$(date -Is) P9 supervisor started; no reward or ranking data consulted"

# P8 owns the first pass over this fresh panel.  A file existence check is a
# scheduling guard only; P9 never reads its score values.
while [[ ! -s $E/stage4_eval/goal_assessment.json ]]; do sleep 60; done
for arm in "${MANDATORY[@]}"; do
  f=$E/stage4/$arm/train/full/job_status.json
  while [[ ! -s $f ]]; do sleep 60; done
  "$VENV/bin/python" - "$f" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
raise SystemExit(0 if d.get("status") == "completed" and d.get("returncode") == 0 and d.get("finite_metrics") else 1)
PY
done

MODELS=(base "${MANDATORY[@]}")
f=$E/stage4/$OPTIONAL/train/full/job_status.json
while [[ ! -s $f ]]; do sleep 60; done
if "$VENV/bin/python" - "$f" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
raise SystemExit(0 if d.get("status") == "completed" and d.get("returncode") == 0 and d.get("finite_metrics") else 1)
PY
then MODELS+=("$OPTIONAL"); else echo "$(date -Is) optional top-mass failed; evaluating mandatory paired replication only"; fi

while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; do sleep 60; done
for i in 1 2 3; do date -Is; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; sleep 10; done

if [[ ! -s $OUT/gates/base.json ]]; then E=$E OUT=$OUT bash "$SCRIPT_DIR/decode_and_gate_seed43_model.sh" base 0; fi
pending=()
for m in "${MODELS[@]:1}"; do [[ -s $OUT/gates/$m.json ]] || pending+=("$m"); done
while ((${#pending[@]})); do
  pids=()
  for gpu in 0 1; do
    ((${#pending[@]})) || break
    m=${pending[0]}; pending=("${pending[@]:1}")
    (E=$E OUT=$OUT bash "$SCRIPT_DIR/decode_and_gate_seed43_model.sh" "$m" "$gpu") & pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p"; done
done

MERGE=$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/merge_eval_pool.py
SHARD=$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py
CSV=$(IFS=,; echo "${MODELS[*]}")
"$VENV/bin/python" "$MERGE" --generation-root "$OUT/generations" --models "$CSV" --seed 42 --expected-records 1000 --gate-root "$OUT/gates" --output "$OUT/response_pool.jsonl" --audit "$OUT/pool_audit.json"
"$VENV/bin/python" "$SHARD" split --input "$OUT/response_pool.jsonl" --output-dir "$OUT/shards" --num-shards 4 --expected-records 1000
for obj in helpfulness harmlessness; do
  for pair in '0:0 1:1' '0:2 1:3'; do
    jobs=()
    for item in $pair; do IFS=: read -r gpu shard <<<"$item"; jobs+=("$gpu:$obj:$shard"); done
    P9OUT=$OUT bash "$SCRIPT_DIR/score_seed43_workers.sh" "${jobs[@]}"
  done
done
count=$("$VENV/bin/python" - "$OUT/pool_audit.json" <<'PY'
import json,sys
print(len(json.load(open(sys.argv[1]))["eligible_models"]))
PY
)
for obj in helpfulness harmlessness; do
  "$VENV/bin/python" "$SHARD" merge --inputs "$OUT"/score_shards/${obj}_{0,1,2,3}.jsonl --output "$OUT/scores/$obj.jsonl" --audit "$OUT/scores/${obj}_audit.json" --expected-records 1000 --expected-scores-per-row "$count" --strip-responses
done
"$VENV/bin/python" "$PROJECT/analysis/p5_8b_robust_stage1_stage2_20260717/aggregate_stage1_comparison.py" \
  --helpfulness "$OUT/scores/helpfulness.jsonl" --harmlessness "$OUT/scores/harmlessness.jsonl" --pool-audit "$OUT/pool_audit.json" --output-dir "$OUT/results" \
  --bootstrap 2000 --seed 42 --scope "P9 optimization-seed-43 replication on P8's already-used 1,000-prompt panel; not a new confirmation and not an independent end-to-end seed" \
  --report-title "P9 Stage-4 optimization-seed-43 replication" --ronpo-arm ronpo_os_stage4_s43 --ronpo-arm ronpo_topmass_stage4_s43 --comparison-label "IPO in the preregistered paired replication"
"$VENV/bin/python" - "$OUT" <<'PY'
import hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1]); summary=out/'results/ranked_validation_summary.json'
payload={
  'status':'completed',
  'p8_primary_read':False,
  'selection_allowed':False,
  'limitation':'optimization-seed replication only; P7 parents and P8 precompute remain seed 42',
  'summary_sha256':hashlib.sha256(summary.read_bytes()).hexdigest(),
  'spent_sealed_split_touched':False,
}
(out/'evaluation_audit.json').write_text(json.dumps(payload,indent=2)+'\n')
PY
echo "$(date -Is) P9 supervisor completed"
