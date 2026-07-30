#!/usr/bin/env bash
# Reward-blind corrected stability gate for one completed P5 Stage-2 arm.
# This is intentionally separate from reward scoring and always uses the
# frozen 49-prompt P4 panel with its explicit record count.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 ARM GPU" >&2
  exit 2
fi
ARM=$1
GPU=$2
PROJECT=${PROJECT:-/NHNHOME/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/AIPR/sjkim/venv_clean}
EXP=${EXP:-$PROJECT/results/p5_8b_robust_stage1_stage2_20260717}
P4=${P4:-$PROJECT/results/p4_8b_saferlhf_table4_20260717}
OUT=$EXP/stage2/$ARM/stability_validation
MANIFEST=$P4/dataset_manifest/validation_conflict.jsonl
EXPECTED=$(wc -l < "$MANIFEST")
[[ "$EXPECTED" == "49" ]] || { echo "unexpected fixed-panel size: $EXPECTED" >&2; exit 1; }
[[ -f "$EXP/stage2/$ARM/train/full/job_status.json" ]] || { echo "full job status missing for $ARM" >&2; exit 1; }
[[ ! -f "$OUT/gate.json" ]] || { echo "refusing to overwrite existing gate: $OUT/gate.json" >&2; exit 1; }
mkdir -p "$OUT/logs" "$OUT/generation"
source "$VENV/bin/activate"
export PYTHONPATH="$PROJECT" VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
export TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1
CUDA_VISIBLE_DEVICES=$GPU "$VENV/bin/python" "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py" \
  --manifest "$MANIFEST" --model "$EXP/stage2/$ARM/train/full" --policy-name "$ARM" --probe none \
  --output "$OUT/generation/output_42.json" --seed 42 --temperature 0.7 --top-p 0.9 --max-new-tokens 512 --max-model-len 8192 --gpu-memory-utilization 0.88 \
  > "$OUT/logs/decode.log" 2>&1
"$VENV/bin/python" "$PROJECT/scripts/revision/flagship/stability_gate_corrected.py" \
  --base "$EXP/stage1/fixed_p4_validation/generations/base/output_42.json" \
  --candidate "$OUT/generation/output_42.json" --output "$OUT/gate.json" --expected-records "$EXPECTED" \
  --min-length-ratio 0.33 --max-length-ratio 2.0 --max-repeat-run 20 > "$OUT/logs/gate.log" 2>&1
