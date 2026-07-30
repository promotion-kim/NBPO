#!/usr/bin/env bash
# Resume extraction and ArmoRM scoring from completed RMOD eval tensors.
# This intentionally never invokes eval.py, so a completed decode is not repeated.
set -euo pipefail
SJ=/NHNHOME/AIPR/sjkim
REPO=$SJ/rmod_20260720/repo
PROJECT=$SJ/MNPO_rev_20260720
WORK=$SJ/rmod_20260720/radar
CACHE=$SJ/baseline_repair_1p5b_20260714/cache/huggingface/hub
VC=$SJ/venv_clean/bin/python
GPU=${GPU:-0}
export HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface
export HF_HUB_CACHE=$CACHE HUGGINGFACE_HUB_CACHE=$CACHE
export TOKENIZERS_PARALLELISM=false PYTHONPATH=$PROJECT

for K in "$@"; do
  TAG=chat_b4_k${K}
  PT=$(find "$REPO/eval_outputs" -name "eval_outputs_rmod_${TAG}_*.pt" -printf '%T@ %p\n' \
    | sort -nr | head -1 | cut -d' ' -f2-)
  test -n "$PT"
  OUT=$WORK/gens/rmod_${TAG}/generated.json
  mkdir -p "$(dirname "$OUT")" "$WORK/scored" "$WORK/gens/logs"
  $VC - "$PT" "$OUT" <<'PY'
import json, os, sys, torch

def raw_prompt(prompt):
    start = "<start_of_turn>user\n" if "<start_of_turn>user\n" in prompt else "user\n"
    end = ("<end_of_turn>\n<start_of_turn>model\n"
           if "<end_of_turn>\n<start_of_turn>model\n" in prompt else "\nmodel\n")
    if start in prompt and end in prompt:
        prompt = prompt.split(start, 1)[1].rsplit(end, 1)[0]
    return prompt.strip()

rows = torch.load(sys.argv[1], weights_only=False)
records = {}
for row in rows:
    records.setdefault(raw_prompt(row["prompt"]), []).append(row["response"])
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump([{"prompt": p, "all_generated_responses": r}
               for p, r in records.items()], handle)
print(f"extracted {len(records)} prompts from {os.path.basename(sys.argv[1])}")
PY
  CUDA_VISIBLE_DEVICES=$GPU $VC -m on_policy_data_gen.rm_armo_multihead \
    --input_file "$OUT" --output_dir "$WORK/scored" \
    --split rmod_${TAG} --indices 6,7,8,9,10 \
    --names instruction_following,truthfulness,honesty,helpfulness,safety \
    --cache_dir "$CACHE" --batch_size 8 --sample_batch_size 32 \
    2>&1 | tee -a "$WORK/gens/logs/rmod_${TAG}_score_resume.log"
  echo "[resume-uf-chat] K=$K done at $(date -Is)"
done
