#!/usr/bin/env bash
# Decode the preregistered P4 union pool exactly once, one independent seed per GPU.
set -euo pipefail

PROJECT=${PROJECT:-/NHNHOME/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/AIPR/sjkim/venv_clean}
ROOT=${ROOT:-/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1}
EXP="$PROJECT/results/p4_8b_saferlhf_kappa_imbalance_20260717"
POOL="$EXP/train_pool"
MODEL="$ROOT/base_objective_screen/hf_ipv4/llama31"
MANIFEST="$EXP/dataset_manifest/union_train_prompts.jsonl"
EXPECTED=8617

mkdir -p "$POOL/generations" "$POOL/logs"
export PYTHONPATH="$PROJECT"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_CUDNN_SDPA_ENABLED=0
export TOKENIZERS_PARALLELISM=false

decode_one() {
  local gpu=$1 seed=$2 name="seed${2}"
  local output="$POOL/generations/$name/output_${seed}.json"
  mkdir -p "$(dirname "$output")"
  if [[ -s "$output" ]] && "$VENV/bin/python" - "$output" "$EXPECTED" <<'PY'
import json, sys
try:
    assert len(json.load(open(sys.argv[1], encoding="utf-8"))) == int(sys.argv[2])
except Exception:
    raise SystemExit(1)
PY
  then
    echo "$(date -Is) skip-complete seed=$seed" | tee -a "$POOL/logs/decode_${name}.log"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$VENV/bin/python" \
    "$PROJECT/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py" \
    --manifest "$MANIFEST" --model "$MODEL" --policy-name "$name" --probe none \
    --output "$output" --seed "$seed" --temperature 0.7 --top-p 0.9 \
    --max-new-tokens 1024 --max-model-len 8192 --gpu-memory-utilization 0.88 \
    >"$POOL/logs/decode_${name}.log" 2>&1
}

pids=()
for index in 0 1 2 3; do
  decode_one "$index" "$((42 + index))" & pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ "$status" -ne 0 ]]; then
  echo "$(date -Is) decode-failed" > "$POOL/DECODE_FAILED"
  exit "$status"
fi
date -Is > "$POOL/DECODE_COMPLETE"
