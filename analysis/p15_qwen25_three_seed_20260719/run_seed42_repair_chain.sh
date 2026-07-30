#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 6 ]] || { echo "usage: $0 PROJECT PY ROOT CACHE ARM GPU" >&2; exit 2; }
PROJECT=$1 PY=$2 ROOT=$3 CACHE=$4 ARM=$5 GPU=$6
case "$ARM" in inpo_avg) CAND=inpo_norm_a;; ipo) CAND=ipo_norm_a;; *) exit 2;; esac
S=$PROJECT/analysis/p14_qwen25_7b_stage4_seeds_20260718
for STAGE in 2 3 4; do
  GATE=$ROOT/seeds/s42/stage$STAGE/gates/$ARM.json
  [[ -s $GATE ]] || bash "$S/decode_and_gate.sh" "$PROJECT" "$PY" "$PY" "$ROOT" 42 "$STAGE" "$ARM" "$GPU"
  "$PY" - "$GATE" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); assert d.get('passed') is True and d.get('status') == 'passed', d
PY
  (( STAGE == 4 )) && break
  NEXT=$((STAGE+1))
  [[ -s $ROOT/seeds/s42/stage$NEXT/$ARM/pool/PREPARED ]] || \
    bash "$S/prepare_continuation_pool.sh" "$PROJECT" "$PY" "$PY" "$ROOT" "$CACHE" 42 "$NEXT" "$ARM" "$GPU"
  "$PY" "$S/train_repaired_stage.py" --project "$PROJECT" --python "$PY" --root "$ROOT" \
    --cache "$CACHE" --lock "$S/baseline_repair_lock.json" --candidate "$CAND" --arm "$ARM" --stage "$NEXT" --gpu "$GPU"
done

