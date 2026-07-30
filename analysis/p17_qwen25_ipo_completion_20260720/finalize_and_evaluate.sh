#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 6 ]] || { echo "usage: $0 PROJECT PY SOURCE OUT CACHE DEADLINE" >&2; exit 2; }
PROJECT=$1 PY=$2 SOURCE=$3 OUT=$4 CACHE=$5 DEADLINE=$6
selected=
for candidate in ipo_repair_a ipo_repair_b ipo_repair_c ipo_repair_d; do
  gate=$OUT/candidate_runs/$candidate/s43/stage4/gates/ipo.json
  while [[ ! -s $gate ]]; do (( $(date +%s) < DEADLINE )) || exit 124; sleep 20; done
  [[ $("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("passed"))' "$gate") == True ]] && { selected=$candidate; break; }
done
[[ -n $selected ]] || { echo "no gate-passing IPO candidate" >&2; exit 1; }
"$PY" - "$OUT" "$selected" <<'PY'
import hashlib,json,sys,time
from pathlib import Path
out=Path(sys.argv[1]); c=sys.argv[2]; g=out/f'candidate_runs/{c}/s43/stage4/gates/ipo.json'
(out/'selection.json').write_text(json.dumps({'status':'selected','candidate':c,'selection_uses_reward':False,
 'selected_at':time.time(),'gate_sha256':hashlib.sha256(g.read_bytes()).hexdigest()},indent=2)+'\n')
PY
AUDIT=$SOURCE/audit_ipo_before_completion_20260720
mkdir -p "$AUDIT/s43"
model=$SOURCE/seeds/s43/stage4/ipo/train/full
gen=$SOURCE/seeds/s43/stage4/generations/ipo
gate=$SOURCE/seeds/s43/stage4/gates/ipo.json
[[ -e $model || -L $model ]] && mv "$model" "$AUDIT/s43/original_train_full"
[[ -e $gen || -L $gen ]] && mv "$gen" "$AUDIT/s43/original_generation"
[[ -e $gate || -L $gate ]] && mv "$gate" "$AUDIT/s43/original_gate.json"
mkdir -p "$(dirname "$model")" "$(dirname "$gen")" "$(dirname "$gate")"
ln -s "$OUT/candidate_runs/$selected/s43/stage4/train/full" "$model"
ln -s "$OUT/candidate_runs/$selected/s43/stage4/generations/ipo" "$gen"
ln -s "$OUT/candidate_runs/$selected/s43/stage4/gates/ipo.json" "$gate"
if [[ -d $SOURCE/evaluation ]]; then mv "$SOURCE/evaluation" "$AUDIT/evaluation_before_ipo"; fi
mkdir -p "$SOURCE/evaluation/logs"
while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; do sleep 20; done
EVAL_GPUS=0,1,2,3 bash "$PROJECT/analysis/p14_qwen25_7b_stage4_seeds_20260718/evaluate_after_training.sh" \
  "$PROJECT" "$PY" "$SOURCE" "$CACHE" > "$SOURCE/evaluation/logs/evaluate.log" 2>&1
cp -a "$SOURCE/evaluation/common_eligible_arms.json" "$SOURCE/evaluation/three_seed" "$OUT/"
date -Is > "$OUT/COMPLETE"
