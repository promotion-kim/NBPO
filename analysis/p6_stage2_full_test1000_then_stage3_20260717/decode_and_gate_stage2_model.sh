#!/usr/bin/env bash
# Decode and reward-blind gate exactly one Stage-2 model on the locked P6 panel.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 MODEL_KEY GPU" >&2
  exit 2
fi
MODEL_KEY=$1
GPU=$2
PROJECT=${PROJECT:-/NHNHOME/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/AIPR/sjkim/venv_clean}
ROOT=${ROOT:-/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1}
P5=${P5:-$PROJECT/results/p5_8b_robust_stage1_stage2_20260717}
P6=${P6:-$PROJECT/results/p6_stage2_full_test1000_then_stage3_20260717}
OUT=$P6/stage2_eval
MANIFEST=${MANIFEST:-$P6/fresh_test_1000.jsonl}
EXPECTED=1000

case "$MODEL_KEY" in
  base) MODEL=$ROOT/base_objective_screen/hf_ipv4/llama31 ;;
  ronpo_os_stage2|ronpo_topmass_stage2|inpo_avg_stage2|sppo_avg_stage2|simpo_stage2|ipo_stage2|dpo_stage2|ht_mnpo_harmless_stage2|ht_mnpo_helpfulness_stage2)
    MODEL=$P5/stage2/$MODEL_KEY/train/full ;;
  *) echo "unknown model key: $MODEL_KEY" >&2; exit 2 ;;
esac
[[ -f "$MANIFEST" && -f "$MODEL/config.json" ]] || { echo "missing locked input/model" >&2; exit 1; }
source "$VENV/bin/activate"
export PYTHONPATH="$PROJECT" VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
export TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1
mkdir -p "$OUT"/{generations,gates,logs}
GEN=$OUT/generations/$MODEL_KEY/output_42.json
CUDA_VISIBLE_DEVICES=$GPU "$VENV/bin/python" "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py" \
  --manifest "$MANIFEST" --model "$MODEL" --policy-name "$MODEL_KEY" --probe none --output "$GEN" \
  --seed 42 --temperature 0.7 --top-p 0.9 --max-new-tokens 512 --max-model-len 8192 --gpu-memory-utilization 0.88 \
  > "$OUT/logs/decode_${MODEL_KEY}_$(hostname).log" 2>&1
BASE=$OUT/generations/base/output_42.json
if [[ "$MODEL_KEY" == base ]]; then BASE=$GEN; fi
set +e
"$VENV/bin/python" "$PROJECT/scripts/revision/flagship/stability_gate_corrected.py" \
  --base "$BASE" --candidate "$GEN" --output "$OUT/gates/$MODEL_KEY.json" --expected-records "$EXPECTED" \
  --min-length-ratio 0.33 --max-length-ratio 2.0 --max-repeat-run 20 \
  > "$OUT/logs/gate_${MODEL_KEY}_$(hostname).log" 2>&1
RC=$?
set -e
if [[ "$MODEL_KEY" == base && $RC -ne 0 ]]; then exit "$RC"; fi
exit 0
