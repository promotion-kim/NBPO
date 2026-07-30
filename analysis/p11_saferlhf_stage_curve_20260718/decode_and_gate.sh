#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 5 ]] || { echo "usage: $0 PROJECT OUTPUT MODEL_KEY MODEL_PATH GPU" >&2; exit 2; }
PROJECT=$1; OUT=$2; KEY=$3; MODEL=$4; GPU=$5
VENV=$PROJECT/../venv_clean
PANEL=$PROJECT/results/p8_stage4_fresh_default_test_20260718/dataset_manifest/fresh_default_test_1000.jsonl
BASE=$OUT/generations/base/output_42.json
GEN=$OUT/generations/$KEY/output_42.json
mkdir -p "$OUT/generations/$KEY" "$OUT/gates" "$OUT/logs/decode"
if [[ ! -s "$GEN" ]]; then
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PROJECT VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false \
    TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1 \
    "$VENV/bin/python" "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py" \
    --manifest "$PANEL" --model "$MODEL" --policy-name "$KEY" --probe none --output "$GEN" \
    --seed 42 --temperature 0.7 --top-p 0.9 --max-new-tokens 512 --max-model-len 8192 --gpu-memory-utilization 0.55 \
    > "$OUT/logs/decode/${KEY}_$(hostname).log" 2>&1
fi
[[ -s "$BASE" ]] || { echo "base generation must be available before gating $KEY" >&2; exit 1; }
set +e
"$VENV/bin/python" "$PROJECT/scripts/revision/flagship/stability_gate_corrected.py" \
  --base "$BASE" --candidate "$GEN" --output "$OUT/gates/$KEY.json" --expected-records 1000 \
  --min-length-ratio 0.33 --max-length-ratio 2.0 --max-repeat-run 20 \
  > "$OUT/logs/decode/gate_${KEY}_$(hostname).log" 2>&1
gate_rc=$?
set -e
# Exit 4 is the detector's documented fail-closed model outcome. Preserve it
# in the gate JSON and continue evaluating the remaining locked checkpoints.
[[ $gate_rc -eq 0 || $gate_rc -eq 4 ]] || exit "$gate_rc"
