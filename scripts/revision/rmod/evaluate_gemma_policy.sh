#!/usr/bin/env bash
set -euo pipefail

TAG=${TAG:?set TAG}
MODEL=${MODEL:?set MODEL}
WORK=${WORK:?set WORK}
GPU=${GPU:?set GPU}
SJ=/NHNHOME/AIPR/sjkim
P=$SJ/MNPO_rev_20260720
PY=$SJ/venv_clean/bin/python
CACHE=$SJ/baseline_repair_1p5b_20260714/cache/huggingface/hub
# Match the 128-token candidate decode below with the existing 128-token
# Base decode. The former 1,024-token Base reference invalidated the ratio.
BASE_GEN=$SJ/rmod_20260720/radar/gens/base128/output_42.json
SCORED=$SJ/rmod_20260720/radar/scored
mkdir -p "$WORK/eval" "$WORK/logs" "$SCORED"
exec 9>"$WORK/eval/.evaluation.lock"
flock 9
[[ ! -s "$WORK/eval/COMPLETE" ]] || exit 0

export PYTHONPATH=$P
export HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface
export HF_HUB_CACHE=$CACHE HUGGINGFACE_HUB_CACHE=$CACHE
export TOKENIZERS_PARALLELISM=false MNPO_DISABLE_APEX=1 NCCL_SOCKET_IFNAME=lo

if [[ ! -s "$WORK/eval/output_42.json" ]]; then
  CUDA_VISIBLE_DEVICES=$GPU "$PY" -u -m on_policy_data_gen.decode \
    --data_dir "$P/data/gemma2_ufb_part2_test.jsonl" --model "$MODEL" --seeds 42 \
    --output_dir "$WORK/eval" --num_gpu 1 --temperature 0.8 --top_p 0.95 \
    --max_tokens 128 --batch_size 256 --dtype bfloat16 --cache_dir "$CACHE" \
    > "$WORK/logs/decode_eval.log" 2>&1
fi

"$PY" "$P/scripts/revision/flagship/stability_gate_corrected.py" \
  --base "$BASE_GEN" --candidate "$WORK/eval/output_42.json" \
  --expected-records 647 --min-length-ratio 0.33 --max-length-ratio 2.0 \
  --max-repeat-run 20 --output "$WORK/eval/stability_gate.json" \
  > "$WORK/logs/stability_gate.log" 2>&1

if [[ ! -s "$WORK/eval/fmt.json" ]]; then
  "$PY" - "$WORK/eval/output_42.json" "$WORK/eval/fmt.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
json.dump([{"prompt": r["prompt"], "all_generated_responses": [r["generated_text"]]} for r in rows], open(sys.argv[2], "w"))
PY
fi

if [[ ! -s "$SCORED/${TAG}_safety.jsonl" ]]; then
  CUDA_VISIBLE_DEVICES=$GPU "$PY" -m on_policy_data_gen.rm_armo_multihead \
    --input_file "$WORK/eval/fmt.json" --output_dir "$SCORED" --split "$TAG" \
    --indices 6,7,8,9,10 --names instruction_following,truthfulness,honesty,helpfulness,safety \
    --cache_dir "$CACHE" --batch_size 16 --sample_batch_size 32 \
    > "$WORK/logs/score_eval.log" 2>&1
fi

date -Is > "$WORK/eval/COMPLETE"
