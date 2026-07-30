#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then echo "usage: $0 MODEL_KEY GPU" >&2; exit 2; fi
MODEL_KEY=$1; GPU=$2
PROJECT=${PROJECT:-/NHNHOME/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/AIPR/sjkim/venv_clean}
ROOT=${ROOT:-/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1}
P7=${P7:?P7 experiment path is required}
MANIFEST=${MANIFEST:?locked P7 manifest is required}
EXPECTED=1000; OUT=$P7/stage3_eval
case "$MODEL_KEY" in
  base) MODEL=$ROOT/base_objective_screen/hf_ipv4/llama31 ;;
  ronpo_os_stage3|ronpo_topmass_stage3|inpo_avg_stage3|sppo_avg_stage3|simpo_stage3|ipo_stage3|dpo_stage3|ht_mnpo_harmless_stage3|ht_mnpo_helpfulness_stage3) MODEL=$P7/stage3/$MODEL_KEY/train/full ;;
  *) echo "unknown model: $MODEL_KEY" >&2; exit 2 ;;
esac
[[ -f "$MANIFEST" && -f "$MODEL/config.json" ]] || { echo "missing locked input/model" >&2; exit 1; }
source "$VENV/bin/activate"
export PYTHONPATH="$PROJECT" VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1
mkdir -p "$OUT"/{generations,gates,logs}
GEN=$OUT/generations/$MODEL_KEY/output_42.json
CUDA_VISIBLE_DEVICES=$GPU "$VENV/bin/python" "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py" --manifest "$MANIFEST" --model "$MODEL" --policy-name "$MODEL_KEY" --probe none --output "$GEN" --seed 42 --temperature 0.7 --top-p 0.9 --max-new-tokens 512 --max-model-len 8192 --gpu-memory-utilization 0.88 > "$OUT/logs/decode_${MODEL_KEY}_$(hostname).log" 2>&1
BASE=$OUT/generations/base/output_42.json; [[ "$MODEL_KEY" == base ]] && BASE=$GEN
set +e
"$VENV/bin/python" "$PROJECT/scripts/revision/flagship/stability_gate_corrected.py" --base "$BASE" --candidate "$GEN" --output "$OUT/gates/$MODEL_KEY.json" --expected-records "$EXPECTED" --min-length-ratio 0.33 --max-length-ratio 2.0 --max-repeat-run 20 > "$OUT/logs/gate_${MODEL_KEY}_$(hostname).log" 2>&1
RC=$?
set -e
[[ "$MODEL_KEY" != base || $RC -eq 0 ]]
