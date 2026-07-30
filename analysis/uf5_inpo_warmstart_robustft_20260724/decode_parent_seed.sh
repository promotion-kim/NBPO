#!/usr/bin/env bash
set -euo pipefail
GPU=${1:?gpu}
SEED=${2:?seed}
TEMP=${3:?temperature}
SJ=/NHNHOME/AIPR/sjkim
P=$SJ/MNPO_rev_20260720
ROOT=$SJ/uf5_inpo_warmstart_robustft_20260724
PY=$SJ/venv_clean/bin/python
CACHE=$SJ/baseline_repair_1p5b_20260714/cache/huggingface/hub
PARENT=$SJ/ronpo_gemma_baselines_s1/inpo
export PYTHONPATH=$P HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface
export HF_HUB_CACHE=$CACHE HUGGINGFACE_HUB_CACHE=$CACHE TOKENIZERS_PARALLELISM=false
for split in train test; do
  expected=19856
  [[ $split == test ]] && expected=647
  out=$ROOT/pool/$split/parent
  mkdir -p "$out" "$ROOT/logs"
  file=$out/output_${SEED}.json
  if [[ -s $file ]] && "$PY" - "$file" "$expected" <<'PY'
import json,sys
assert len(json.load(open(sys.argv[1]))) == int(sys.argv[2])
PY
  then
    continue
  fi
  CUDA_VISIBLE_DEVICES=$GPU "$PY" -u -m on_policy_data_gen.decode \
    --data_dir "$P/data/gemma2_ufb_part2_${split}.jsonl" --model "$PARENT" \
    --seeds "$SEED" --output_dir "$out" --num_gpu 1 --temperature "$TEMP" \
    --top_p 0.95 --max_tokens 1024 --batch_size 512 --dtype bfloat16 \
    --cache_dir "$CACHE" > "$ROOT/logs/decode_${split}_parent_${SEED}.log" 2>&1
  "$PY" - "$file" "$expected" <<'PY'
import json,sys
assert len(json.load(open(sys.argv[1]))) == int(sys.argv[2])
PY
done
date -Is > "$ROOT/pool/PARENT_SEED_${SEED}_COMPLETE"

