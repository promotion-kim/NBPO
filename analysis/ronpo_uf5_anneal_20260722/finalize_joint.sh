#!/usr/bin/env bash
set -euo pipefail
SJ=/NHNHOME/AIPR/sjkim; P=$SJ/MNPO_rev_20260720; ROOT=$SJ/ronpo_uf5_anneal_20260722
PY=$SJ/venv_clean/bin/python
terminal() { [[ -s "$ROOT/$1/ARM_COMPLETE" ]] || find "$ROOT/$1" -name GATE_FAILED -type f -print -quit 2>/dev/null | grep -q .; }
while ! terminal moving_anchor || ! terminal stronger_signal; do sleep 30; done
mkdir -p "$ROOT/joint" "$ROOT/joint/logs" "$ROOT/results"
models=(
  "base=Base=$SJ/rmod_20260720/radar/gens/base128/output_42.json"
  "rmod=RMOD K=16=$SJ/rmod_20260720/radar/gens/rmod_chat_b4_k16/generated.json"
  "locked_s1=RONPO S1=$SJ/ronpo_gemma_20260720/kappa_arms/os_k0p05/gen/output_42.json"
  "locked_s2=RONPO S2=$SJ/ronpo_gemma_s2/eval/output_42.json"
  "locked_s3=RONPO S3=$SJ/ronpo_gemma_s3/eval/output_42.json"
  "locked_s4=RONPO S4=$SJ/ronpo_gemma_s4/eval/output_42.json"
  "locked_s5=RONPO S5=$SJ/ronpo_gemma_s5/eval/output_42.json"
)
for arm in moving_anchor stronger_signal; do
  short=MA; [[ $arm == stronger_signal ]] && short=SS
  for stage in 1 2 3 4; do
    gate=$ROOT/$arm/stage$stage/eval/stability_gate.json; gen=$ROOT/$arm/stage$stage/eval/output_42.json
    if [[ -s $gate && -s $gen ]] && "$PY" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("passed") is True else 1)' "$gate"; then
      models+=("${arm}_s${stage}=RONPO-${short}-S${stage}=$gen")
    fi
  done
done
"$PY" "$P/analysis/ronpo_uf5_anneal_20260722/build_joint_pool.py" \
  --prompt-manifest "$ROOT/frozen_prompt_manifest_586.json" --models "${models[@]}" \
  --output "$ROOT/joint/joint_pool.json" --audit "$ROOT/joint/pool_audit.json"
date -Is > "$ROOT/joint/joint_pool.READY"
if [[ ! -s "$ROOT/joint/SCORES_0_3_COMPLETE" ]]; then
  bash "$P/analysis/ronpo_uf5_anneal_20260722/score_worker.sh" 0 3 > "$ROOT/joint/logs/worker_0_3.log" 2>&1
fi
while [[ ! -s "$ROOT/joint/SCORES_4_7_COMPLETE" ]]; do sleep 30; done
"$PY" "$P/analysis/ronpo_uf5_anneal_20260722/merge_joint_scores.py" \
  --scores "$ROOT/joint/scores" --expected-count 586 \
  --audit "$ROOT/joint/jsonl_validation.json"
"$PY" "$P/analysis/ronpo_uf5_anneal_20260722/aggregate_joint_scores.py" \
  --scores "$ROOT/joint/scores" --pool-audit "$ROOT/joint/pool_audit.json" \
  --out "$ROOT/results" --bootstrap 2000 --seed 42
"$PY" "$P/analysis/rmod_validation_20260721/plot_uf_stage_extension.py" \
  --summary "$ROOT/results/paired_summary.json" --out "$ROOT/results/uf5_anneal_trajectory.pdf"
"$PY" "$P/analysis/ronpo_uf5_anneal_20260722/write_report.py" \
  --summary "$ROOT/results/paired_summary.json" --out "$ROOT/REPORT.md" --root "$ROOT"
"$PY" "$P/analysis/ronpo_uf5_anneal_20260722/publish_hf_results.py" --root "$ROOT"
date -Is > "$ROOT/SCORING_COMPLETE"
"$PY" "$P/analysis/ronpo_uf5_anneal_20260722/prune_verified.py" --root "$ROOT"
find "$ROOT" -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' -o -name '*.pdf' -o -name '*.png' \) -print0 | sort -z | xargs -0 sha256sum > "$ROOT/ARTIFACT_SHA256.txt"
date -Is > "$ROOT/COMPLETE"
