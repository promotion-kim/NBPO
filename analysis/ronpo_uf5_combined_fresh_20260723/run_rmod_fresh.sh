#!/usr/bin/env bash
set -euo pipefail
SJ=/NHNHOME/AIPR/sjkim
REPO=$SJ/rmod_20260720/repo
P=$SJ/MNPO_rev_20260720
ROOT=$SJ/ronpo_uf5_combined_fresh_20260723
CACHE=$SJ/baseline_repair_1p5b_20260714/cache/huggingface/hub
VR=$SJ/rmod_20260720/venv_rmod/bin/python
VC=$SJ/venv_clean/bin/python
DATA=$P/data/gemma2_ufb_part3_test.jsonl
WORK=$ROOT/fresh/rmod
TAG=uf5_combined_fresh_b4_k16
CKPT=$SJ/rmod_20260720/repo/outputs/2026-07-21/15-35-28/uf_vf/gemma2_2b_5head_20260721_153529/1784615738/transformer-step=14993-validation_iql_loss=0.00.ckpt
export HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface
export HF_HUB_CACHE=$CACHE HUGGINGFACE_HUB_CACHE=$CACHE TOKENIZERS_PARALLELISM=false PYTHONPATH=$P
export RMOD_UF_DATA=$WORK/uf_eval_ds_gemma_chat RMOD_CKPT=$CKPT
mkdir -p "$WORK/gens/logs" "$ROOT/audit"

test "$(sha256sum "$CKPT" | cut -d' ' -f1)" = d952bfddd992e5ffc0c4bcf90c3108d12f4c70c728c683d936daf69ddfd97cfe
if [[ ! -d "$RMOD_UF_DATA" ]]; then
  "$VC" "$P/scripts/revision/rmod/build_uf_eval_dataset.py" --prompts_jsonl "$DATA" \
    --output_dir "$RMOD_UF_DATA" --chat_template_model google/gemma-2-2b-it
fi
cd "$REPO"
CUDA_VISIBLE_DEVICES=0 "$VR" eval.py +experiment=rmod_uf_radar \
  model=gemma_2_2b_multi_head_5 \
  ++dataset._target_=scripts.revision.rmod.all_multiobjective_dataset.AllMultiObjectiveDataset \
  ++dataset.data_path="$RMOD_UF_DATA" ++trainer.devices=[0] \
  ++decoder.tree_depth=4 ++decoder.num_branches=16 ++decoder.lambda_coef=0.5 \
  ++experiment_name="rmod_${TAG}" 2>&1 | tee "$WORK/gens/logs/rmod_${TAG}.log"
PT=$(find "$REPO/eval_outputs" -name "eval_outputs_rmod_${TAG}_*.pt" -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
"$VC" - "$PT" "$WORK/gens/generated.json" <<'PY'
import json,sys,torch

def raw_prompt(text):
    start="<start_of_turn>user\n" if "<start_of_turn>user\n" in text else "user\n"
    end="<end_of_turn>\n<start_of_turn>model\n" if "<end_of_turn>\n<start_of_turn>model\n" in text else "\nmodel\n"
    if start in text and end in text: text=text.split(start,1)[1].rsplit(end,1)[0]
    return text.strip()

rows={}
for row in torch.load(sys.argv[1],weights_only=False):
    rows.setdefault(raw_prompt(row['prompt']),[]).append(row['response'])
assert len(rows)==647, len(rows)
json.dump([{'prompt':p,'all_generated_responses':ys} for p,ys in rows.items()],open(sys.argv[2],'w'))
PY
sha256sum "$WORK/gens/generated.json" > "$ROOT/audit/rmod_fresh_sha256.txt"
date -Is > "$ROOT/fresh/RMOD_COMPLETE"
