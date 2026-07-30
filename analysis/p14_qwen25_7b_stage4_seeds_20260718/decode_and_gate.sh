#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 8 ]] || { echo "usage: $0 PROJECT TRAIN_PY INFER_PY ROOT SEED STAGE ARM GPU" >&2; exit 2; }
PROJECT=$1; TRAIN_PY=$2; INFER_PY=$3; ROOT=$4; SEED=$5; STAGE=$6; ARM=$7; GPU=$8
MODEL=$ROOT/seeds/s$SEED/stage$STAGE/$ARM/train/full
MANIFEST=$ROOT/dataset_manifest/fresh_default_test_1000.jsonl
BASE=$ROOT/shared/eval/base/output_42.json
OUT=$ROOT/seeds/s$SEED/stage$STAGE
GEN=$OUT/generations/$ARM/output_42.json
GATE=$OUT/gates/$ARM.json
[[ -f $MODEL/config.json && -s $MANIFEST && -s $BASE ]] || { echo "gate prerequisites missing" >&2; exit 1; }
mkdir -p "$(dirname "$GEN")" "$(dirname "$GATE")" "$OUT/logs"
export PYTHONPATH=$PROJECT VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
export TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1
CUDA_VISIBLE_DEVICES=$GPU "$INFER_PY" "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py" \
  --manifest "$MANIFEST" --model "$MODEL" --policy-name "q25_s${SEED}_stage${STAGE}_${ARM}" --probe none \
  --output "$GEN" --seed 42 --temperature 0.7 --top-p 0.9 --max-new-tokens 512 --max-model-len 8192 \
  --gpu-memory-utilization 0.88 > "$OUT/logs/decode_${ARM}.log" 2>&1
"$TRAIN_PY" "$PROJECT/scripts/revision/flagship/stability_gate_corrected.py" \
  --base "$BASE" --candidate "$GEN" --output "$GATE" --expected-records 1000 \
  --min-length-ratio 0.33 --max-length-ratio 2.0 --max-repeat-run 20 > "$OUT/logs/gate_${ARM}.log" 2>&1

