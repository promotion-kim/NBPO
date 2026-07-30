#!/usr/bin/env bash
set -euo pipefail

SJ=/NHNHOME/AIPR/sjkim
PROJECT=$SJ/MNPO_rev_20260720
PY=$SJ/venv_clean/bin/python
SCORED=$SJ/rmod_20260720/radar/scored
UF_OUT=$SJ/rmod_20260720/validated/uf_chat_b4
SAFE=$SJ/rmod_20260720/saferlhf_ksweep_chat_b16_n1000
SAFE_OUT=$SJ/rmod_20260720/validated/saferlhf_chat
mkdir -p "$UF_OUT" "$SAFE_OUT"

while [[ $(find "$SCORED" -maxdepth 1 -type f \
  \( -name 'rmod_chat_b4_k1_*.jsonl' -o -name 'rmod_chat_b4_k16_*.jsonl' \) \
  | wc -l) -lt 10 ]]; do
  sleep 60
done

$PY "$PROJECT/analysis/rmod_validation_20260721/validate_uf_rmod.py" \
  --scored-dir "$SCORED" \
  --methods rmod_chat_b4_k1=K1_reference ronpo_gemma_k0p05=RONPO \
    rmod_chat_b4_k16=RMOD \
  --reference K1_reference --out-dir "$UF_OUT" \
  > "$UF_OUT/driver.log" 2>&1
$PY "$PROJECT/analysis/rmod_validation_20260721/plot_paired_uf_radar.py" \
  --summary "$UF_OUT/paired_summary.json" \
  --out "$UF_OUT/uf5_radar.pdf" \
  >> "$UF_OUT/driver.log" 2>&1
$PY "$PROJECT/analysis/rmod_validation_20260721/audit_generations.py" \
  --inputs \
    K1_reference="$SJ/rmod_20260720/radar/gens/rmod_chat_b4_k1/generated.json" \
    RMOD="$SJ/rmod_20260720/radar/gens/rmod_chat_b4_k16/generated.json" \
  --reference K1_reference --out "$UF_OUT/generation_audit.json" \
  >> "$UF_OUT/driver.log" 2>&1

until $PY "$PROJECT/analysis/rmod_validation_20260721/collect_saferlhf_ksweep.py" \
  --summary-dir "$SAFE" --out "$SAFE_OUT/rmod_k.jsonl" \
  > "$SAFE_OUT/driver.log" 2>&1; do
  sleep 60
done
$PY "$PROJECT/analysis/rmod_validation_20260721/audit_generations.py" \
  --inputs \
    K1_reference="$SAFE/k1_fmt.jsonl" K2="$SAFE/k2_fmt.jsonl" \
    K4="$SAFE/k4_fmt.jsonl" K8="$SAFE/k8_fmt.jsonl" K16="$SAFE/k16_fmt.jsonl" \
  --reference K1_reference --out "$SAFE_OUT/generation_audit.json" \
  >> "$SAFE_OUT/driver.log" 2>&1
$PY "$PROJECT/scripts/revision/rmod/plot_saferlhf_stage_front.py" \
  --rmod "$SAFE_OUT/rmod_k.jsonl" --out "$SAFE_OUT/saferlhf_stage_front.pdf" \
  >> "$SAFE_OUT/driver.log" 2>&1
echo "POSTPROCESS_DONE=$(date -Is)"
