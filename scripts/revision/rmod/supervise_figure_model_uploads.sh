#!/usr/bin/env bash
# Upload only gate-passing figure checkpoints, then prune weights after final evaluation.
set -euo pipefail
SJ=/NHNHOME/AIPR/sjkim
P10=$SJ/MNPO_rev_20260710
P20=$SJ/MNPO_rev_20260720
PY=$SJ/venv_clean/bin/python
AUDIT=$SJ/rmod_stage5_kappa_large_20260722/hf_uploads
mkdir -p "$AUDIT"

gate_passed() {
  "$PY" - "$1" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); raise SystemExit(0 if x.get("passed") is True and x.get("status") == "passed" else 1)
PY
}

for stage in 3 4 5; do
  [[ -s "$AUDIT/figure2_stage${stage}.json" ]] && continue
  gate=$SJ/ronpo_gemma_s${stage}/eval/stability_gate.json
  while [[ ! -s "$gate" ]]; do sleep 60; done
  gate_passed "$gate" || { echo "Figure 2 Stage $stage failed gate; not uploading" >> "$AUDIT/failures.log"; continue; }
  "$PY" "$P20/scripts/revision/rmod/upload_figure_checkpoint.py" \
    --repo promotion/ronpo-gemma2-2b-uf5-stage3-5-s42 \
    --model "$SJ/ronpo_gemma_s${stage}/stage${stage}" --path-in-repo "stage${stage}" \
    --audit "$AUDIT/figure2_stage${stage}.json" --title "RONPO Gemma-2-2B UltraFeedback Stage 3-5"
done

F3=$P10/results/rmod_figure3_kappa_large_20260722
while [[ ! -s "$F3/eval_joint/COMPLETED" ]]; do sleep 60; done
for label in k1 k2; do
  for stage in 1 2 3 4; do
    [[ -s "$AUDIT/figure3_${label}_stage${stage}.json" ]] && continue
    if [[ $stage -eq 1 ]]; then
      model=$F3/$label/stage1/full; gate=$F3/eval_joint/gates/${label}_stage1.json
    else
      model=$F3/$label/stage${stage}_run/stage${stage}/ronpo_os_${label}/train/full
      gate=$F3/$label/stage${stage}_run/stage${stage}_stability_p8_locked_panel/gates/ronpo_os_${label}.json
    fi
    gate_passed "$gate" || { echo "Figure 3 $label Stage $stage failed gate; not uploading" >> "$AUDIT/failures.log"; continue; }
    "$PY" "$P20/scripts/revision/rmod/upload_figure_checkpoint.py" \
      --repo promotion/ronpo-llama31-8b-saferlhf-kappa1-2-stage1-4-s42 \
      --model "$model" --path-in-repo "$label/stage${stage}" \
      --audit "$AUDIT/figure3_${label}_stage${stage}.json" --title "RONPO Llama-3.1-8B SafeRLHF large-kappa stages"
  done
done

while [[ ! -s "$SJ/ronpo_gemma_final_20260722/COMPLETED" ]]; do sleep 60; done
for stage in 3 4 5; do
  find "$SJ/ronpo_gemma_s${stage}" -type d -name 'checkpoint-*' -prune -exec rm -rf {} +
  find "$SJ/ronpo_gemma_s${stage}/smoke20" -maxdepth 1 -type f -name '*.safetensors' -delete 2>/dev/null || true
  find "$SJ/ronpo_gemma_s${stage}/stage${stage}" -maxdepth 1 -type f -name '*.safetensors' -delete
done
for label in k1 k2; do
  for stage in 1 2 3 4; do
    if [[ $stage -eq 1 ]]; then model=$F3/$label/stage1/full; else model=$F3/$label/stage${stage}_run/stage${stage}/ronpo_os_${label}/train/full; fi
    find "$model" -maxdepth 1 -type f -name '*.safetensors' -delete
  done
done
date -Is > "$AUDIT/UPLOADS_VERIFIED_AND_LOCAL_WEIGHTS_PRUNED"
