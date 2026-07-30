#!/usr/bin/env bash
set -euo pipefail

SPLIT=${1:?usage: run_p2_hh_eval.sh validation|fresh}
PROJECT=${PROJECT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/venv_clean}
EXP=$PROJECT/results/p2_8b_hh_multiobjective_20260717
ANALYSIS=$PROJECT/analysis/p2_8b_hh_multiobjective_20260717
ROOT=/NHNHOME/26msit001_A/BASE/AIPR/sjkim/revision_qwen3_8b/full_iter1
BASE=$ROOT/base_objective_screen/hf_ipv4/llama31
WILD=$(tail -1 "$PROJECT/results/p1_8b_base_objective_screen_20260716/downloads/wildguard.path" | sed -n 's/^  path: //p; t; p')
GUARD=$ROOT/flagship_20260712/cache/huggingface/models--Qwen--Qwen3Guard-Gen-8B/snapshots/4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb
MODELS=base,ronpo_os,ronpo_topmass,inpo_avg,ht_mnpo_harmless,ht_mnpo_helpful,sppo_avg,simpo,ipo,ronpo_full_expect
if [[ "$SPLIT" == validation ]]; then
  MANIFEST=$PROJECT/results/p1_8b_base_objective_screen_20260716/dataset_manifest/validation.jsonl
  EXPECTED=640
  EXPECTED_SHA=1d21a88603095af464fa64a057736c4717ec4ac69a377ffe1720378f19319ce9
elif [[ "$SPLIT" == fresh ]]; then
  MANIFEST=$PROJECT/results/p1_8b_base_objective_screen_20260716/dataset_manifest/fresh_confirmation.jsonl
  EXPECTED=320
  EXPECTED_SHA=2bf999f00c26bdfeb2bf73a97844dcb13395dc8eef9e65ab03c22f531a14f3d2
  [[ -s "$EXP/validation/model_summary.json" ]] || { echo "validation table is not final" >&2; exit 9; }
  mkdir -p "$EXP/audit"
  if [[ ! -s "$EXP/audit/FRESH_OPENED.json" ]]; then
    python - "$EXP/audit/FRESH_OPENED.json" "$MANIFEST" "$EXPECTED_SHA" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
p=Path(sys.argv[2]); observed=hashlib.sha256(p.read_bytes()).hexdigest()
assert observed == sys.argv[3]
Path(sys.argv[1]).write_text(json.dumps({"opened_at":datetime.now(timezone.utc).astimezone().isoformat(),"manifest":str(p),"sha256":observed,"purpose":"single locked fresh primary confirmation"},indent=2)+"\n")
PY
  fi
else
  echo "unknown split $SPLIT" >&2; exit 2
fi
[[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" == "$EXPECTED_SHA" ]]

ER=$EXP/$SPLIT
mkdir -p "$ER/generations" "$ER/logs" "$ER/gates" "$ER/shards" "$ER/score_shards" "$ER/scores"
source "$VENV/bin/activate"
export PYTHONPATH=$PROJECT
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_CUDNN_SDPA_ENABLED=0
export WANDB_MODE=disabled

: >"$ER/logs/prelaunch_gpu_samples.txt"
for sample in 1 2 3; do
  date -Is >>"$ER/logs/prelaunch_gpu_samples.txt"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader >>"$ER/logs/prelaunch_gpu_samples.txt"
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader)
  printf '%s\n' "$apps" >>"$ER/logs/prelaunch_gpu_samples.txt"
  [[ -z "$apps" ]] || { echo "compute process present before $SPLIT evaluation; fail closed" >&2; exit 8; }
  [[ "$sample" -eq 3 ]] || sleep 2
done

decode_one() {
  local gpu=$1 model_name=$2 model_path=$3
  mkdir -p "$ER/generations/$model_name"
  CUDA_VISIBLE_DEVICES=$gpu python "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py" \
    --manifest "$MANIFEST" --model "$model_path" --policy-name "$model_name" --probe none \
    --output "$ER/generations/$model_name/output_42.json" --seed 42 \
    --temperature 0.7 --top-p 0.9 --max-new-tokens 1024 \
    >"$ER/logs/decode_${model_name}.log" 2>&1
}

worker0() { decode_one 0 base "$BASE"; decode_one 0 ht_mnpo_harmless "$EXP/train/full/ht_mnpo_harmless/checkpoint-900"; decode_one 0 ipo "$EXP/train/stretch/full/ipo/checkpoint-900"; }
worker1() { decode_one 1 ronpo_os "$EXP/train/full/ronpo_os/checkpoint-900"; decode_one 1 ht_mnpo_helpful "$EXP/train/stretch/full/ht_mnpo_helpful/checkpoint-900"; decode_one 1 ronpo_full_expect "$EXP/train/stretch/full/ronpo_full_expect/checkpoint-900"; }
worker2() { decode_one 2 ronpo_topmass "$EXP/train/full/ronpo_topmass/checkpoint-900"; decode_one 2 sppo_avg "$EXP/train/stretch/full/sppo_avg/checkpoint-900"; }
worker3() { decode_one 3 inpo_avg "$EXP/train/full/inpo_avg/checkpoint-900"; decode_one 3 simpo "$EXP/train/stretch/full/simpo/checkpoint-900"; }
worker0 & p0=$!
worker1 & p1=$!
worker2 & p2=$!
worker3 & p3=$!
wait "$p0" "$p1" "$p2" "$p3"

for model in ${MODELS//,/ }; do
  python "$PROJECT/scripts/revision/flagship/stability_gate_corrected.py" \
    --base "$ER/generations/base/output_42.json" \
    --candidate "$ER/generations/$model/output_42.json" \
    --output "$ER/gates/$model.json" --expected-records "$EXPECTED" \
    --min-length-ratio 0.33 --max-length-ratio 2.0 --max-repeat-run 20 \
    >"$ER/logs/gate_${model}.log" 2>&1 || true
  [[ -s "$ER/gates/$model.json" ]]
done

python "$ANALYSIS/merge_eval_pool.py" \
  --generation-root "$ER/generations" --models "$MODELS" --seed 42 \
  --expected-records "$EXPECTED" --gate-root "$ER/gates" \
  --output "$ER/response_pool.jsonl" --audit "$ER/pool_audit.json" \
  >"$ER/logs/merge_eval_pool.log" 2>&1
python "$ANALYSIS/shard_score_input.py" split --input "$ER/response_pool.jsonl" \
  --output-dir "$ER/shards" --num-shards 2 --expected-records "$EXPECTED" \
  >"$ER/logs/shard_score_input.log" 2>&1

score_wild() {
  local gpu=$1 shard=$2
  CUDA_VISIBLE_DEVICES=$gpu python "$PROJECT/on_policy_data_gen/rm_wildguard_compliance.py" \
    --model_name "$WILD" --model_revision "$(basename "$WILD")" \
    --input_file "$ER/shards/shard_${shard}.jsonl" --output_file "$ER/score_shards/helpfulness_${shard}.jsonl" \
    --batch_size 8 --sample_batch_size 2 --max_seq_length 4096 \
    >"$ER/logs/helpfulness_${shard}.log" 2>&1
}
score_guard() {
  local gpu=$1 shard=$2
  CUDA_VISIBLE_DEVICES=$gpu python "$PROJECT/on_policy_data_gen/rm_guard_logodds.py" \
    --model_name "$GUARD" --family qwen3guard --model_revision 4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb \
    --input_file "$ER/shards/shard_${shard}.jsonl" --output_file "$ER/score_shards/harmlessness_${shard}.jsonl" \
    --batch_size 8 --sample_batch_size 2 --max_seq_length 4096 \
    >"$ER/logs/harmlessness_${shard}.log" 2>&1
}
score_wild 0 0 & s0=$!; score_wild 1 1 & s1=$!
score_guard 2 0 & s2=$!; score_guard 3 1 & s3=$!
wait "$s0" "$s1" "$s2" "$s3"

python "$ANALYSIS/shard_score_input.py" merge \
  --inputs "$ER/score_shards/helpfulness_0.jsonl" "$ER/score_shards/helpfulness_1.jsonl" \
  --output "$ER/scores/helpfulness.jsonl" --audit "$ER/scores/helpfulness_audit.json" \
  --expected-records "$EXPECTED" --expected-scores-per-row "$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["eligible_models"]))' "$ER/pool_audit.json")" --strip-responses \
  >"$ER/logs/merge_helpfulness.log" 2>&1
python "$ANALYSIS/shard_score_input.py" merge \
  --inputs "$ER/score_shards/harmlessness_0.jsonl" "$ER/score_shards/harmlessness_1.jsonl" \
  --output "$ER/scores/harmlessness.jsonl" --audit "$ER/scores/harmlessness_audit.json" \
  --expected-records "$EXPECTED" --expected-scores-per-row "$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["eligible_models"]))' "$ER/pool_audit.json")" --strip-responses \
  >"$ER/logs/merge_harmlessness.log" 2>&1

primary_flag=()
[[ "$SPLIT" == fresh ]] && primary_flag=(--primary-only)
python "$ANALYSIS/build_two_objective_table.py" \
  --helpfulness "$ER/scores/helpfulness.jsonl" --harmlessness "$ER/scores/harmlessness.jsonl" \
  --pool-audit "$ER/pool_audit.json" --gate-root "$ER/gates" --output-dir "$ER" \
  --split "$SPLIT" --expected-records "$EXPECTED" --bootstrap-resamples 2000 --seed 42 \
  "${primary_flag[@]}" \
  >"$ER/logs/build_table.log" 2>&1

# Scores and gate JSONs now contain every metric input; remove redundant raw text copies.
rm -rf "$ER/generations" "$ER/shards" "$ER/score_shards" "$ER/response_pool.jsonl"
date -Is >"$ER/EVAL_COMPLETE"
if [[ "$SPLIT" == validation ]]; then
  cp "$ER/TABLE.md" "$EXP/TABLE.md"
  cp "$ER/table_two_objective.tex" "$EXP/table_two_objective.tex"
  cp "$ER/GATE.md" "$EXP/GATE.md"
else
  cp "$ER/FRESH.md" "$EXP/FRESH.md"
fi
