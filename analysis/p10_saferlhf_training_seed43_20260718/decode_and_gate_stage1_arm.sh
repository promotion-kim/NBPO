#!/usr/bin/env bash
# Decode exactly one finished P10 seed-43 Stage-1 arm on the P8 locked panel.
# This is a pre-registered held-out optimizer-seed comparison, not model selection.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 ARM GPU" >&2; exit 2; }

ARM=$1
GPU=$2
PROJECT=${PROJECT:-/NHNHOME/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/AIPR/sjkim/venv_clean}
ROOT=${ROOT:-/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1}
E=${E:-$PROJECT/results/p10_saferlhf_training_seed43_20260718}
P8=${P8:-$PROJECT/results/p8_stage4_fresh_default_test_20260718}
MANIFEST=$P8/dataset_manifest/fresh_default_test_1000.jsonl
OUT=$E/stage1_eval_p8_locked_panel
EXPECTED=1000

case "$ARM" in
  ronpo_os|ronpo_topmass|inpo_avg|sppo_avg|simpo|ipo|dpo|ht_mnpo_harmless|ht_mnpo_helpfulness)
    MODEL=$E/stage1/$ARM/train/full ;;
  *) echo "unknown P10 arm: $ARM" >&2; exit 2 ;;
esac
[[ -f "$MODEL/config.json" && -s "$MANIFEST" ]] || { echo "missing completed model or locked manifest" >&2; exit 1; }

source "$VENV/bin/activate"
export PYTHONPATH="$PROJECT" VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
export TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1
mkdir -p "$OUT"/{generations,gates,logs}
ln -sfn "$P8/stage4_eval/generations/base" "$OUT/generations/base"
GEN=$OUT/generations/$ARM/output_43.json
CUDA_VISIBLE_DEVICES=$GPU "$VENV/bin/python" "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py" \
  --manifest "$MANIFEST" --model "$MODEL" --policy-name "p10_stage1_${ARM}_s43" --probe none --output "$GEN" \
  --seed 42 --temperature 0.7 --top-p 0.9 --max-new-tokens 512 --max-model-len 8192 --gpu-memory-utilization 0.88 \
  > "$OUT/logs/decode_${ARM}_$(hostname).log" 2>&1
set +e
"$VENV/bin/python" "$PROJECT/scripts/revision/flagship/stability_gate_corrected.py" \
  --base "$OUT/generations/base/output_42.json" --candidate "$GEN" --output "$OUT/gates/${ARM}.json" --expected-records "$EXPECTED" \
  --min-length-ratio 0.33 --max-length-ratio 2.0 --max-repeat-run 20 \
  > "$OUT/logs/gate_${ARM}_$(hostname).log" 2>&1
RC=$?
set -e
exit "$RC"
