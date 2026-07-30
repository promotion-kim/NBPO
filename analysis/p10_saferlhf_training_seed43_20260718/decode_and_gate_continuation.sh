#!/usr/bin/env bash
# Reward-blind full-panel stability gate for a completed seed-43 continuation model.
set -euo pipefail
[[ $# -eq 5 ]] || { echo "usage: $0 PROJECT EXP STAGE ARM GPU" >&2; exit 2; }
PROJECT=$1; EXP=$2; STAGE=$3; ARM=$4; GPU=$5
VENV=$PROJECT/../venv_clean
P8=$PROJECT/results/p8_stage4_fresh_default_test_20260718
MANIFEST=$P8/dataset_manifest/fresh_default_test_1000.jsonl
BASE=$P8/stage4_eval/generations/base/output_42.json
MODEL=$EXP/$STAGE/$ARM/train/full
OUT=$EXP/${STAGE}_stability_p8_locked_panel
EXPECTED=1000
[[ -f "$MODEL/config.json" && -s "$MANIFEST" && -s "$BASE" ]] || { echo "missing model, manifest, or frozen base generation" >&2; exit 1; }
source "$VENV/bin/activate"
export PYTHONPATH="$PROJECT" VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
export TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1
mkdir -p "$OUT"/{generations,gates,logs}
TRAIN_SEED=${TRAIN_SEED:-43}
GEN=$OUT/generations/$ARM/output_${TRAIN_SEED}.json
exec 9>"$OUT/.evaluation.lock"
flock 9
if [[ -s "$GEN" && -s "$OUT/gates/$ARM.json" ]]; then
  exit 0
fi
CUDA_VISIBLE_DEVICES=$GPU "$VENV/bin/python" "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py" \
  --manifest "$MANIFEST" --model "$MODEL" --policy-name "${RUN_PREFIX:-p10}_${STAGE}_${ARM}_s${TRAIN_SEED}" --probe none --output "$GEN" \
  --seed 42 --temperature 0.7 --top-p 0.9 --max-new-tokens 512 --max-model-len 8192 --gpu-memory-utilization 0.55 \
  > "$OUT/logs/decode_${ARM}_$(hostname).log" 2>&1
"$VENV/bin/python" "$PROJECT/scripts/revision/flagship/stability_gate_corrected.py" \
  --base "$BASE" --candidate "$GEN" --output "$OUT/gates/$ARM.json" --expected-records "$EXPECTED" \
  --min-length-ratio 0.33 --max-length-ratio 2.0 --max-repeat-run 20 \
  > "$OUT/logs/gate_${ARM}_$(hostname).log" 2>&1
