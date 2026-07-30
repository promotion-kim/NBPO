#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 9 ]] || { echo "usage: $0 PROJECT PY SOURCE OUTPUT_ROOT CANDIDATE SEED MODEL GPU CACHE" >&2; exit 2; }
PROJECT=$1 PY=$2 SOURCE=$3 OUT=$4 CAND=$5 SEED=$6 MODEL=$7 GPU=$8 CACHE=$9
MANIFEST=$SOURCE/dataset_manifest/fresh_default_test_1000.jsonl
BASE=$SOURCE/shared/eval/base/output_42.json
ROOT=$OUT/$CAND/s$SEED/stage4
GEN=$ROOT/generations/sppo_avg/output_42.json
GATE=$ROOT/gates/sppo_avg.json
mkdir -p "$(dirname "$GEN")" "$(dirname "$GATE")" "$ROOT/logs"
export PYTHONPATH=$PROJECT HF_HOME=$CACHE VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
export TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1
CUDA_VISIBLE_DEVICES=$GPU "$PY" "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py" \
  --manifest "$MANIFEST" --model "$MODEL" --policy-name "q25_s${SEED}_stage4_sppo_${CAND}" --probe none \
  --output "$GEN" --seed 42 --temperature 0.7 --top-p 0.9 --max-new-tokens 512 \
  --max-model-len 8192 --gpu-memory-utilization 0.88 > "$ROOT/logs/decode.log" 2>&1
"$PY" "$PROJECT/scripts/revision/flagship/stability_gate_corrected.py" --base "$BASE" --candidate "$GEN" \
  --output "$GATE" --expected-records 1000 --min-length-ratio 0.33 --max-length-ratio 2.0 \
  --max-repeat-run 20 > "$ROOT/logs/gate.log" 2>&1

