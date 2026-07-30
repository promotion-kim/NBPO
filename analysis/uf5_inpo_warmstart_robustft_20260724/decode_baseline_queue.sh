#!/usr/bin/env bash
set -euo pipefail
GPU=${1:?gpu}
shift
SJ=/NHNHOME/AIPR/sjkim
P=$SJ/MNPO_rev_20260720
ROOT=$SJ/uf5_inpo_warmstart_robustft_20260724
PY=$SJ/venv_clean/bin/python
CACHE=$SJ/baseline_repair_1p5b_20260714/cache/huggingface/hub
export PYTHONPATH=$P HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface
export HF_HUB_CACHE=$CACHE HUGGINGFACE_HUB_CACHE=$CACHE TOKENIZERS_PARALLELISM=false
for spec in "$@"; do
  tag=${spec%%=*}
  model=${spec#*=}
  out=$ROOT/eval/$tag
  mkdir -p "$out" "$ROOT/logs"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" -u -m on_policy_data_gen.decode \
    --data_dir "$ROOT/dataset_manifest/fresh_part1_test_646.jsonl" \
    --model "$model" --seeds 42 --output_dir "$out" --num_gpu 1 \
    --temperature 0.8 --top_p 0.95 --max_tokens 128 --batch_size 256 \
    --dtype bfloat16 --cache_dir "$CACHE" \
    > "$ROOT/logs/decode_fresh_${tag}.log" 2>&1
  "$PY" - "$out/output_42.json" <<'PY'
import json,sys
assert len(json.load(open(sys.argv[1]))) == 646
PY
  date -Is > "$out/DECODE_COMPLETE"
done

