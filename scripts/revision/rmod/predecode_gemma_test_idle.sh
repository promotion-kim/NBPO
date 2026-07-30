#!/usr/bin/env bash
# Fill GPUs released by train-pool decode with the already-locked test seeds.
set -euo pipefail
SOURCE=${SOURCE:?set SOURCE policy}
WORK=${WORK:?set WORK}
SJ=/NHNHOME/AIPR/sjkim
P=$SJ/MNPO_rev_20260720
PY=$SJ/venv_clean/bin/python
CACHE=$SJ/baseline_repair_1p5b_20260714/cache/huggingface/hub
export PYTHONPATH=$P TOKENIZERS_PARALLELISM=false

valid() {
  [[ -s "$1" ]] && "$PY" - "$1" "$2" <<'PY'
import json, sys
raise SystemExit(0 if len(json.load(open(sys.argv[1]))) == int(sys.argv[2]) else 1)
PY
}

run_one() {
  local gpu=$1 seed=$2
  local train=$WORK/pool/train/gpu$gpu/output_${seed}.json
  local test=$WORK/pool/test/gpu$gpu/output_${seed}.json
  while ! valid "$train" 19856; do sleep 2; done
  valid "$test" 647 && return
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u -m on_policy_data_gen.decode \
    --data_dir "$P/data/gemma2_ufb_part2_test.jsonl" --model "$SOURCE" --seeds "$seed" \
    --output_dir "$WORK/pool/test/gpu$gpu" --num_gpu 1 --temperature 0.8 --top_p 0.95 \
    --max_tokens 1024 --batch_size 512 --dtype bfloat16 --cache_dir "$CACHE" \
    > "$WORK/pool/logs/predecode_test_gpu${gpu}.log" 2>&1
}

run_one 1 42 & p1=$!
run_one 2 79 & p2=$!
run_one 3 100 & p3=$!
wait "$p1" "$p2" "$p3"
date -Is > "$WORK/pool/TEST_PREDECODE_IDLE_COMPLETE"
