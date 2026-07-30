#!/usr/bin/env bash
set -euo pipefail
SJ=/NHNHOME/AIPR/sjkim; P=$SJ/MNPO_rev_20260720; ROOT=$SJ/ronpo_uf5_combined_fresh_20260723
PY=$SJ/venv_clean/bin/python; CACHE=$SJ/baseline_repair_1p5b_20260714/cache/huggingface/hub
PARENT=$ROOT/parents/ss_s4/stronger_signal/stage4
TRAIN=$ROOT/shared/stage5/pool/train/parent_standard
OUT=$ROOT/shared/stage5/pool/test
export PYTHONPATH=$P HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface HF_HUB_CACHE=$CACHE HUGGINGFACE_HUB_CACHE=$CACHE
while true; do
  ready=1
  for seed in 100 101 102 103; do [[ -s "$TRAIN/output_${seed}.json" ]] || ready=0; done
  (( ready == 1 )) && break
  sleep 10
done
mkdir -p "$OUT/parent_standard" "$OUT/parent_hot" "$ROOT/shared/stage5/logs"
decode() {
  local seeds=$1 temp=$2 out=$3 gpu=$4 tag=$5
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u -m on_policy_data_gen.decode \
    --data_dir "$P/data/gemma2_ufb_part2_test.jsonl" --model "$PARENT" --seeds $seeds \
    --output_dir "$out" --num_gpu 1 --temperature "$temp" --top_p 0.95 --max_tokens 1024 \
    --batch_size 512 --dtype bfloat16 --cache_dir "$CACHE" > "$ROOT/shared/stage5/logs/prefetch_${tag}.log" 2>&1
}
decode '100 102' 0.8 "$OUT/parent_standard" 2 standard_even & p0=$!
decode '101 103' 0.8 "$OUT/parent_standard" 3 standard_odd & p1=$!
wait "$p0"; wait "$p1"
decode 104 1.0 "$OUT/parent_hot" 2 hot_104 & p0=$!
decode 105 1.0 "$OUT/parent_hot" 3 hot_105 & p1=$!
wait "$p0"; wait "$p1"
date -Is > "$OUT/PREFETCH_COMPLETE"
