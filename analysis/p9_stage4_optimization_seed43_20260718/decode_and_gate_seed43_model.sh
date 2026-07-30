#!/usr/bin/env bash
# Decode one preregistered P9 optimization-seed arm on the already-locked P8
# panel.  This script never reads the spent 604-prompt split.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 MODEL_KEY GPU" >&2; exit 2; }

MODEL_KEY=$1
GPU=$2
PROJECT=${PROJECT:-/NHNHOME/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/AIPR/sjkim/venv_clean}
ROOT=${ROOT:-/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1}
E=${E:-$PROJECT/results/p8_stage4_fresh_default_test_20260718}
OUT=${OUT:-$E/p9_seed43_eval}
MANIFEST=$E/dataset_manifest/fresh_default_test_1000.jsonl
EXPECTED=1000

case "$MODEL_KEY" in
  base) MODEL=$ROOT/base_objective_screen/hf_ipv4/llama31 ;;
  ronpo_os_stage4_s43|ipo_stage4_s43|ronpo_topmass_stage4_s43)
    MODEL=$E/stage4/$MODEL_KEY/train/full ;;
  *) echo "unknown P9 model: $MODEL_KEY" >&2; exit 2 ;;
esac
[[ -s $MANIFEST && -f $MODEL/config.json ]] || { echo "missing locked model or manifest" >&2; exit 1; }

source "$VENV/bin/activate"
export PYTHONPATH="$PROJECT" VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
export TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1
mkdir -p "$OUT"/{generations,gates,logs}
GEN=$OUT/generations/$MODEL_KEY/output_42.json
CUDA_VISIBLE_DEVICES=$GPU "$VENV/bin/python" "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py" \
  --manifest "$MANIFEST" --model "$MODEL" --policy-name "$MODEL_KEY" --probe none --output "$GEN" \
  --seed 42 --temperature 0.7 --top-p 0.9 --max-new-tokens 512 --max-model-len 8192 --gpu-memory-utilization 0.88 \
  >"$OUT/logs/decode_${MODEL_KEY}_$(hostname).log" 2>&1

BASE=$OUT/generations/base/output_42.json
[[ $MODEL_KEY == base ]] && BASE=$GEN
set +e
"$VENV/bin/python" "$PROJECT/scripts/revision/flagship/stability_gate_corrected.py" \
  --base "$BASE" --candidate "$GEN" --output "$OUT/gates/$MODEL_KEY.json" --expected-records "$EXPECTED" \
  --min-length-ratio 0.33 --max-length-ratio 2.0 --max-repeat-run 20 \
  >"$OUT/logs/gate_${MODEL_KEY}_$(hostname).log" 2>&1
RC=$?
set -e
[[ $MODEL_KEY != base || $RC -eq 0 ]]
