#!/usr/bin/env bash
set -euo pipefail
SJ=/NHNHOME/AIPR/sjkim
P=$SJ/MNPO_rev_20260720
PY=$SJ/venv_clean/bin/python
SCORED=$SJ/rmod_20260720/radar/scored
OUT=$SJ/ronpo_gemma_final_20260722
mkdir -p "$OUT"
terminal_eval() { [[ -s "$1/eval/COMPLETE" ]]; }
while ! terminal_eval "$SJ/ronpo_gemma_s2"; do sleep 30; done
if grep -q '"status": "passed"' "$SJ/ronpo_gemma_s2/eval/stability_gate.json"; then
  while ! terminal_eval "$SJ/ronpo_gemma_s3"; do sleep 30; done
  if grep -q '"status": "passed"' "$SJ/ronpo_gemma_s3/eval/stability_gate.json"; then
    while ! terminal_eval "$SJ/ronpo_gemma_s4"; do sleep 30; done
    if grep -q '"status": "passed"' "$SJ/ronpo_gemma_s4/eval/stability_gate.json"; then
      while ! terminal_eval "$SJ/ronpo_gemma_s5"; do sleep 30; done
    fi
  fi
fi
methods=(base128=Base rmod_chat_b4_k16=RMOD ronpo_gemma_k0p05=RONPO-S1)
for stage in 2 3 4 5; do
  root=$SJ/ronpo_gemma_s${stage}
  [[ -s "$root/eval/COMPLETE" ]] && methods+=("ronpo_gemma_s${stage}=RONPO-S${stage}")
done
"$PY" "$P/analysis/rmod_validation_20260721/validate_uf_rmod.py" \
  --scored-dir "$SCORED" --methods "${methods[@]}" \
  --reference Base --out-dir "$OUT" --bootstrap 2000 --seed 42 > "$OUT/finalize.log" 2>&1
"$PY" "$P/scripts/revision/rmod/plot_uf_stage_baselines_radar.py" --summary "$OUT/paired_summary.json" --out "$OUT/uf5_radar.pdf" >> "$OUT/finalize.log" 2>&1
date -Is > "$OUT/COMPLETED"
