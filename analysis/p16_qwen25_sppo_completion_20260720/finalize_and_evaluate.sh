#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 6 ]] || { echo "usage: $0 PROJECT PY SOURCE OUTPUT CACHE DEADLINE_EPOCH" >&2; exit 2; }
PROJECT=$1 PY=$2 SOURCE=$3 OUT=$4 CACHE=$5 DEADLINE=$6
RUNS=$OUT/candidate_runs
mkdir -p "$OUT/launch"
selected=
for candidate in sppo_strong_a sppo_strong_b sppo_strong_c; do
  while :; do
    (( $(date +%s) < DEADLINE )) || exit 124
    ready=1
    for seed in 42 43 44; do [[ -s $RUNS/$candidate/s$seed/stage4/gates/sppo_avg.json ]] || ready=0; done
    (( ready )) && break
    sleep 30
  done
  passed=1
  for seed in 42 43 44; do
    [[ $("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("passed"))' "$RUNS/$candidate/s$seed/stage4/gates/sppo_avg.json") == True ]] || passed=0
  done
  if (( passed )); then selected=$candidate; break; fi
done
[[ -n $selected ]] || { echo "no all-seed gate passer" >&2; exit 1; }
"$PY" - "$OUT" "$selected" <<'PY'
import hashlib,json,sys,time
from pathlib import Path
out=Path(sys.argv[1]); c=sys.argv[2]
gates={str(s):out/f'candidate_runs/{c}/s{s}/stage4/gates/sppo_avg.json' for s in (42,43,44)}
payload={'status':'selected','candidate':c,'selection_uses_reward':False,'selected_at':time.time(),
         'gate_sha256':{s:hashlib.sha256(p.read_bytes()).hexdigest() for s,p in gates.items()}}
(out/'selection.json').write_text(json.dumps(payload,indent=2)+'\n')
PY
AUDIT=$SOURCE/audit_sppo_before_completion_20260720
mkdir -p "$AUDIT"
for seed in 42 43 44; do
  mkdir -p "$AUDIT/s$seed"
  model=$SOURCE/seeds/s$seed/stage4/sppo_avg/train/full
  gen=$SOURCE/seeds/s$seed/stage4/generations/sppo_avg
  gate=$SOURCE/seeds/s$seed/stage4/gates/sppo_avg.json
  [[ -e $model || -L $model ]] && mv "$model" "$AUDIT/s$seed/original_train_full"
  [[ -e $gen || -L $gen ]] && mv "$gen" "$AUDIT/s$seed/original_generation"
  [[ -e $gate || -L $gate ]] && mv "$gate" "$AUDIT/s$seed/original_gate.json"
  mkdir -p "$(dirname "$model")" "$(dirname "$gen")" "$(dirname "$gate")"
  ln -s "$RUNS/$selected/s$seed/stage4/train/full" "$model"
  ln -s "$RUNS/$selected/s$seed/stage4/generations/sppo_avg" "$gen"
  ln -s "$RUNS/$selected/s$seed/stage4/gates/sppo_avg.json" "$gate"
done
if [[ -d $SOURCE/evaluation ]]; then mv "$SOURCE/evaluation" "$AUDIT/evaluation_before_sppo"; fi
mkdir -p "$SOURCE/evaluation/logs"
while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; do sleep 20; done
: > "$SOURCE/evaluation/logs/prelaunch_gpu_samples.txt"
for sample in 1 2 3; do
  date -Is >> "$SOURCE/evaluation/logs/prelaunch_gpu_samples.txt"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> "$SOURCE/evaluation/logs/prelaunch_gpu_samples.txt"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader >> "$SOURCE/evaluation/logs/prelaunch_gpu_samples.txt"
  (( sample == 3 )) || sleep 4
done
EVAL_GPUS=0,1,2,3 bash "$PROJECT/analysis/p14_qwen25_7b_stage4_seeds_20260718/evaluate_after_training.sh" \
  "$PROJECT" "$PY" "$SOURCE" "$CACHE" > "$SOURCE/evaluation/logs/evaluate.log" 2>&1
cp -a "$SOURCE/evaluation/common_eligible_arms.json" "$SOURCE/evaluation/three_seed" "$OUT/"
date -Is > "$OUT/COMPLETE"
