#!/usr/bin/env bash
set -euo pipefail

SJ=/NHNHOME/AIPR/sjkim
P=$SJ/MNPO_rev_20260720
PY=$SJ/venv_clean/bin/python
SCORED=$SJ/rmod_20260720/radar/scored
OUT=$SJ/rmod_stage_extension_20260721/final
mkdir -p "$OUT"

for tag in ronpo_gemma_s2 ronpo_gemma_s3 ronpo_gemma_s4; do
  test "$(find "$SCORED" -maxdepth 1 -name "${tag}_*.jsonl" | wc -l)" -eq 5
done

"$PY" "$P/analysis/rmod_validation_20260721/validate_uf_rmod.py" \
  --scored-dir "$SCORED" --methods \
  rmod_chat_b4_k1='Base (K=1)' rmod_chat_b4_k16='RMOD (K=16)' \
  ronpo_gemma_k0p05='RONPO S1' ronpo_gemma_s2='RONPO S2' \
  ronpo_gemma_s3='RONPO S3' ronpo_gemma_s4='RONPO S4' \
  --reference 'Base (K=1)' --out-dir "$OUT" --bootstrap 2000 --seed 42 \
  > "$OUT/build.log" 2>&1
"$PY" "$P/analysis/rmod_validation_20260721/plot_uf_stage_extension.py" \
  --summary "$OUT/paired_summary.json" --out "$OUT/uf5_stage_radar.pdf" \
  >> "$OUT/build.log" 2>&1
date -Is > "$OUT/COMPLETE"
