#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 9 ]] || { echo "usage: $0 PROJECT TRAIN_PY INFER_PY REPAIR_ROOT SOURCE_ROOT CACHE CANDIDATE GPU EXPECTED_STAGE" >&2; exit 2; }
PROJECT=$1; TRAIN_PY=$2; INFER_PY=$3; REPAIR_ROOT=$4; SOURCE_ROOT=$5; CACHE=$6; CANDIDATE=$7; GPU=$8; STAGE=$9
MODEL=$REPAIR_ROOT/candidates/$CANDIDATE/train/full
MANIFEST=$SOURCE_ROOT/dataset_manifest/fresh_default_test_1000.jsonl
BASE=$SOURCE_ROOT/shared/eval/base/output_42.json
GEN=$REPAIR_ROOT/candidates/$CANDIDATE/generation/output_42.json
GATE=$REPAIR_ROOT/candidates/$CANDIDATE/gate.json
[[ -f $MODEL/config.json && -s $MANIFEST && -s $BASE ]] || { echo "gate prerequisites missing" >&2; exit 1; }
mkdir -p "$(dirname "$GEN")" "$REPAIR_ROOT/candidates/$CANDIDATE/logs"
export PYTHONPATH=$PROJECT HF_HOME=$CACHE VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
export TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1
CUDA_VISIBLE_DEVICES=$GPU "$INFER_PY" "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py" \
  --manifest "$MANIFEST" --model "$MODEL" --policy-name "q25_s42_stage${STAGE}_${CANDIDATE}" --probe none \
  --output "$GEN" --seed 42 --temperature 0.7 --top-p 0.9 --max-new-tokens 512 --max-model-len 8192 \
  --gpu-memory-utilization 0.88 > "$REPAIR_ROOT/candidates/$CANDIDATE/logs/decode.log" 2>&1
"$TRAIN_PY" "$PROJECT/scripts/revision/flagship/stability_gate_corrected.py" \
  --base "$BASE" --candidate "$GEN" --output "$GATE" --expected-records 1000 \
  --min-length-ratio 0.33 --max-length-ratio 2.0 --max-repeat-run 20 \
  > "$REPAIR_ROOT/candidates/$CANDIDATE/logs/gate.log" 2>&1
