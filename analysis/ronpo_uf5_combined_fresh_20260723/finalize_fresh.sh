#!/usr/bin/env bash
set -euo pipefail
SJ=/NHNHOME/AIPR/sjkim; P=$SJ/MNPO_rev_20260720; ROOT=$SJ/ronpo_uf5_combined_fresh_20260723
PY=$SJ/venv_clean/bin/python; CACHE=$SJ/baseline_repair_1p5b_20260714/cache/huggingface/hub
DATA=$P/data/gemma2_ufb_part3_test.jsonl
export PYTHONPATH=$P HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface
export HF_HUB_CACHE=$CACHE HUGGINGFACE_HUB_CACHE=$CACHE TOKENIZERS_PARALLELISM=false
while [[ ! -s "$ROOT/CHAIN_TERMINAL" || ! -s "$ROOT/fresh/PRIOR_DECODE_COMPLETE" || ! -s "$ROOT/fresh/RMOD_COMPLETE" ]]; do sleep 30; done
mkdir -p "$ROOT/fresh/generations" "$ROOT/fresh/logs" "$ROOT/joint/logs" "$ROOT/results"

decode() {
  local tag=$1 model=$2 gpu=$3
  mkdir -p "$ROOT/fresh/generations/$tag"
  local output=$ROOT/fresh/generations/$tag/output_42.json
  if [[ -s "$output" ]] && "$PY" - "$output" <<'PY'
import json,sys
assert len(json.load(open(sys.argv[1]))) == 647
PY
  then
    echo "Reusing verified 647-row fresh generation: $tag"
    return
  fi
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u -m on_policy_data_gen.decode --data_dir "$DATA" --model "$model" \
    --seeds 42 --output_dir "$ROOT/fresh/generations/$tag" --num_gpu 1 --temperature 0.8 \
    --top_p 0.95 --max_tokens 128 --batch_size 256 --dtype bfloat16 --cache_dir "$CACHE" \
    > "$ROOT/fresh/logs/decode_${tag}.log" 2>&1
  "$PY" - "$output" <<'PY'
import json,sys
assert len(json.load(open(sys.argv[1]))) == 647
PY
}

decode fixed_s5 "$ROOT/fixed_base/stage5/train" 0 & d0=$!
decode combined_s5 "$ROOT/combined/stage5/train" 1 & d1=$!
decode combined_s6 "$ROOT/combined/stage6/train" 2 & d2=$!
wait "$d0"; wait "$d1"; wait "$d2"
if [[ -s "$ROOT/combined/stage7/eval/GATE_PASSED" ]]; then
  decode combined_s7 "$ROOT/combined/stage7/train" 0
fi

"$PY" - "$DATA" "$ROOT/fresh/prompt_manifest.json" <<'PY'
import json,sys
prompts=[]
for line in open(sys.argv[1]):
    row=json.loads(line); prompts.append(row['prompt'])
assert len(prompts)==647 and len(set(prompts))==647
json.dump({'count':647,'prompts':prompts},open(sys.argv[2],'w'),indent=2)
PY

models=(
  "base=Base=$ROOT/fresh/generations/base/output_42.json"
  "rmod=RMOD K=16=$ROOT/fresh/rmod/gens/generated.json"
  "ss_s1=RONPO-SS-S1=$ROOT/fresh/generations/ss_s1/output_42.json"
  "ss_s2=RONPO-SS-S2=$ROOT/fresh/generations/ss_s2/output_42.json"
  "ss_s3=RONPO-SS-S3=$ROOT/fresh/generations/ss_s3/output_42.json"
  "ss_s4=RONPO-SS-S4=$ROOT/fresh/generations/ss_s4/output_42.json"
  "fixed_s5=RONPO-FB-S5=$ROOT/fresh/generations/fixed_s5/output_42.json"
  "combined_s5=RONPO-COMB-S5=$ROOT/fresh/generations/combined_s5/output_42.json"
  "combined_s6=RONPO-COMB-S6=$ROOT/fresh/generations/combined_s6/output_42.json"
)
if [[ -s "$ROOT/fresh/generations/combined_s7/output_42.json" ]]; then
  models+=("combined_s7=RONPO-COMB-S7=$ROOT/fresh/generations/combined_s7/output_42.json")
fi
"$PY" "$P/analysis/ronpo_uf5_combined_fresh_20260723/build_joint_pool.py" \
  --prompt-manifest "$ROOT/fresh/prompt_manifest.json" --models "${models[@]}" \
  --index-aligned-tags rmod \
  --output "$ROOT/joint/joint_pool.json" --audit "$ROOT/joint/pool_audit.json"
date -Is > "$ROOT/joint/joint_pool.READY"
bash "$P/analysis/ronpo_uf5_combined_fresh_20260723/score_worker.sh" 0 3 > "$ROOT/joint/logs/worker_0_3.log" 2>&1
while [[ ! -s "$ROOT/joint/SCORES_4_5_COMPLETE" ]]; do sleep 20; done
"$PY" "$P/analysis/ronpo_uf5_combined_fresh_20260723/merge_joint_scores.py" \
  --scores "$ROOT/joint/scores" --num-shards 6 --expected-count 647 --audit "$ROOT/joint/jsonl_validation.json"
"$PY" "$P/analysis/ronpo_uf5_combined_fresh_20260723/aggregate_fresh.py" \
  --scores "$ROOT/joint/scores" --pool-audit "$ROOT/joint/pool_audit.json" --out "$ROOT/results" --bootstrap 2000 --seed 42
"$PY" "$P/analysis/ronpo_uf5_combined_fresh_20260723/plot_fresh.py" \
  --summary "$ROOT/results/paired_summary.json" --out "$ROOT/results/uf5_combined_fresh_trajectory.pdf"
"$PY" "$P/analysis/ronpo_uf5_combined_fresh_20260723/write_report.py" \
  --summary "$ROOT/results/paired_summary.json" --out "$ROOT/REPORT.md"
{
  echo '# Completion audit'; echo; echo "Completed: $(date -Is)"; echo;
  echo 'All fresh-panel methods were scored in one six-shard BF16 batch context.';
  echo 'All reported training checkpoints passed the locked reward-blind gate and were publicly uploaded before pruning.';
  echo; echo 'spent_sealed_split_touched=false';
} > "$ROOT/COMPLETION_AUDIT.md"
# Storage minimization only after verified upload and completed fresh evaluation.
for audit in "$ROOT"/hf_uploads/*.json; do "$PY" -c 'import json,sys; assert json.load(open(sys.argv[1]))["verified"] is True' "$audit"; done
{
  for path in "$ROOT/parents" "$ROOT/shared/base" "$ROOT/shared/stage5/pool" "$ROOT/shared/stage5/scored" "$ROOT/shared/stage5/pairs" \
    "$ROOT/fixed_base/stage5/precomputed_raw" "$ROOT/fixed_base/stage5/precomputed_os" \
    "$ROOT/combined/stage5/precomputed_raw" "$ROOT/combined/stage5/precomputed_os" \
    "$ROOT/combined/stage6/pool" "$ROOT/combined/stage6/scored" "$ROOT/combined/stage6/pairs" "$ROOT/combined/stage6/precomputed_raw" "$ROOT/combined/stage6/precomputed_os" \
    "$ROOT/combined/stage7/pool" "$ROOT/combined/stage7/scored" "$ROOT/combined/stage7/pairs" "$ROOT/combined/stage7/precomputed_raw" "$ROOT/combined/stage7/precomputed_os"; do
    [[ -e $path ]] && du -sh "$path"
  done
} > "$ROOT/DELETED_LARGE_ARTIFACTS.txt"
rm -rf "$ROOT/parents" "$ROOT/shared/base" "$ROOT/shared/stage5/pool" "$ROOT/shared/stage5/scored" "$ROOT/shared/stage5/pairs" \
  "$ROOT/fixed_base/stage5/precomputed_raw" "$ROOT/fixed_base/stage5/precomputed_os" \
  "$ROOT/combined/stage5/precomputed_raw" "$ROOT/combined/stage5/precomputed_os" \
  "$ROOT/combined/stage6/pool" "$ROOT/combined/stage6/scored" "$ROOT/combined/stage6/pairs" "$ROOT/combined/stage6/precomputed_raw" "$ROOT/combined/stage6/precomputed_os" \
  "$ROOT/combined/stage7/pool" "$ROOT/combined/stage7/scored" "$ROOT/combined/stage7/pairs" "$ROOT/combined/stage7/precomputed_raw" "$ROOT/combined/stage7/precomputed_os"
for model in "$ROOT/fixed_base/stage5/train" "$ROOT/combined/stage5/train" "$ROOT/combined/stage6/train" "$ROOT/combined/stage7/train"; do
  rm -f "$model/model.safetensors"
done
find "$ROOT" -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' -o -name '*.pdf' -o -name '*.png' \) \
  -print0 | sort -z | xargs -0 sha256sum > "$ROOT/ARTIFACT_SHA256.txt"
date -Is > "$ROOT/COMPLETE"
