#!/usr/bin/env bash
# Upload already gate-passing Figure 3 checkpoints while later stages train.
set -euo pipefail
SJ=/NHNHOME/AIPR/sjkim
P10=$SJ/MNPO_rev_20260710
P20=$SJ/MNPO_rev_20260720
PY=$SJ/venv_clean/bin/python
F3=$P10/results/rmod_figure3_kappa_large_20260722
AUDIT=$SJ/rmod_stage5_kappa_large_20260722/hf_uploads
mkdir -p "$AUDIT"

for label in k1 k2; do
  for stage in 1 2 3; do
    audit=$AUDIT/figure3_${label}_stage${stage}.json
    [[ -s "$audit" ]] && continue
    if [[ $stage -eq 1 ]]; then
      model=$F3/$label/stage1/full
      gate=$F3/eval_joint/gates/${label}_stage1.json
    else
      model=$F3/$label/stage2_run/stage2/ronpo_os_${label}/train/full
      gate=$F3/$label/stage2_run/stage2_stability_p8_locked_panel/gates/ronpo_os_${label}.json
    fi
    "$PY" - "$gate" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
raise SystemExit(0 if x.get("passed") is True and x.get("status") == "passed" else 1)
PY
    "$PY" "$P20/scripts/revision/rmod/upload_figure_checkpoint.py" \
      --repo promotion/ronpo-llama31-8b-saferlhf-kappa1-2-stage1-4-s42 \
      --model "$model" --path-in-repo "$label/stage${stage}" \
      --audit "$audit" --title "RONPO Llama-3.1-8B SafeRLHF large-kappa stages"
  done
done
date -Is > "$AUDIT/FIGURE3_STAGE1_2_UPLOADS_COMPLETE"
