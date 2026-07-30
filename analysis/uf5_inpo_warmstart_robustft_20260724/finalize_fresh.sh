#!/usr/bin/env bash
set -euo pipefail
SJ=/NHNHOME/AIPR/sjkim
P=$SJ/MNPO_rev_20260720
ROOT=$SJ/uf5_inpo_warmstart_robustft_20260724
PY=$SJ/venv_clean/bin/python
while [[ ! -s "$ROOT/EVAL_GENERATIONS_COMPLETE" ]]; do sleep 30; done
for tag in dpo ipo sppo simpo; do
  while [[ ! -s "$ROOT/eval/$tag/DECODE_COMPLETE" ]]; do sleep 30; done
done
if [[ -s "$ROOT/eval/inpo/GATE_FAILED" || -s "$ROOT/eval/ronpo/GATE_FAILED" ]]; then
  printf '# Result\n\nAt least one preregistered arm failed the reward-blind stability gate. No reward comparison was computed.\n' > "$ROOT/REPORT.md"
  date -Is > "$ROOT/COMPLETE"
  exit 0
fi
mkdir -p "$ROOT/joint" "$ROOT/results" "$ROOT/logs"
"$PY" "$P/analysis/ronpo_uf5_anneal_20260722/build_joint_pool.py" \
  --prompt-manifest "$ROOT/dataset_manifest/fresh_prompt_manifest_646.json" \
  --models \
  "base=Base=$ROOT/eval/base/output_42.json" \
  "parent=INPO-S1-PARENT=$ROOT/eval/parent/output_42.json" \
  "dpo=DPO=$ROOT/eval/dpo/output_42.json" \
  "ipo=IPO=$ROOT/eval/ipo/output_42.json" \
  "sppo=SPPO=$ROOT/eval/sppo/output_42.json" \
  "simpo=SimPO=$ROOT/eval/simpo/output_42.json" \
  "control=INPO-CONTROL-S2=$ROOT/eval/inpo/output_42.json" \
  "ronpo=RONPO-FT-S2=$ROOT/eval/ronpo/output_42.json" \
  --output "$ROOT/joint/joint_pool.json" --audit "$ROOT/joint/pool_audit.json"
date -Is > "$ROOT/joint/joint_pool.READY"
bash "$P/analysis/uf5_inpo_warmstart_robustft_20260724/score_fresh_worker.sh" 0 3 &
p0=$!
while [[ ! -s "$ROOT/joint/SCORES_4_5_COMPLETE" ]]; do sleep 30; done
wait "$p0"
"$PY" "$P/analysis/ronpo_uf5_combined_fresh_20260723/merge_joint_scores.py" \
  --scores "$ROOT/joint/scores" --num-shards 6 --expected-count 646 \
  --audit "$ROOT/joint/jsonl_validation.json"
"$PY" "$P/analysis/uf5_inpo_warmstart_robustft_20260724/aggregate_fresh.py" \
  --scores "$ROOT/joint/scores" --pool-audit "$ROOT/joint/pool_audit.json" \
  --out "$ROOT/results" --bootstrap 2000 --seed 42
"$PY" "$P/analysis/uf5_inpo_warmstart_robustft_20260724/write_report.py" \
  --summary "$ROOT/results/paired_summary.json" --output "$ROOT/REPORT.md"
"$PY" "$P/analysis/uf5_inpo_warmstart_robustft_20260724/plot_fresh_radar.py" \
  --summary "$ROOT/results/paired_summary.json" --output "$ROOT/results/uf5_warmstart_radar.pdf"
date -Is > "$ROOT/COMPLETE"
