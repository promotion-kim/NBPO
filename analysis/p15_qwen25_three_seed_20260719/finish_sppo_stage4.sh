#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 5 ]] || { echo "usage: $0 PROJECT PY ROOT CACHE GPU" >&2; exit 2; }
PROJECT=$1 PY=$2 ROOT=$3 CACHE=$4 GPU=$5
S=$PROJECT/analysis/p14_qwen25_7b_stage4_seeds_20260718
while [[ ! -s $ROOT/sppo_extension/candidates/sppo_strong_a/gate.json || ! -s $ROOT/sppo_extension/candidates/sppo_strong_b/gate.json ]]; do sleep 15; done
CAND=$($PY - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1])/'sppo_extension/candidates'
for c in ('sppo_strong_a','sppo_strong_b'):
 d=json.load(open(r/c/'gate.json'))
 if d.get('passed') is True and d.get('status') == 'passed': print(c); break
else: raise SystemExit(1)
PY
)
PARENT=$ROOT/seeds/s42/stage3/sppo_avg/train/full
mkdir -p "$(dirname "$PARENT")" "$ROOT/seeds/s42/stage3/gates"
ln -sfn "$ROOT/sppo_extension/candidates/$CAND/train/full" "$PARENT"
ln -sfn "$ROOT/sppo_extension/candidates/$CAND/gate.json" "$ROOT/seeds/s42/stage3/gates/sppo_avg.json"
bash "$S/prepare_continuation_pool.sh" "$PROJECT" "$PY" "$PY" "$ROOT" "$CACHE" 42 4 sppo_avg "$GPU"
$PY "$S/train_repaired_stage.py" --project "$PROJECT" --python "$PY" --root "$ROOT" --cache "$CACHE" \
  --lock "$S/baseline_repair_extension_lock.json" --candidate "$CAND" --arm sppo_avg --stage 4 --gpu "$GPU"
bash "$S/decode_and_gate.sh" "$PROJECT" "$PY" "$PY" "$ROOT" 42 4 sppo_avg "$GPU"
