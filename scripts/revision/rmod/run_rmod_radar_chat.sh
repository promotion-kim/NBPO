#!/usr/bin/env bash
# Tokenizer-correct RMOD Figure-4 validation. The reference and RMOD paths use
# the same Gemma chat-formatted inputs and differ only in candidate count K.
set -euo pipefail
SJ=/NHNHOME/AIPR/sjkim
REPO=$SJ/rmod_20260720/repo
PROJECT=$SJ/MNPO_rev_20260720
WORK=$SJ/rmod_20260720/radar
CACHE=$SJ/baseline_repair_1p5b_20260714/cache/huggingface/hub
VR=$SJ/rmod_20260720/venv_rmod/bin/python
VC=$SJ/venv_clean/bin/python
GPU=${GPU:-0}; K=${K:-16}; BLOCK=${BLOCK:-4}; LAMBDA=${LAMBDA:-0.5}
TAG=${TAG:-chat_b${BLOCK}_k${K}}
export HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface
export HF_HUB_CACHE=$CACHE HUGGINGFACE_HUB_CACHE=$CACHE
export TOKENIZERS_PARALLELISM=false PYTHONPATH=$PROJECT
export RMOD_UF_DATA=$WORK/uf_eval_ds_gemma_chat
mkdir -p $WORK/gens/logs $WORK/scored

if [[ ! -d $RMOD_UF_DATA ]]; then
  $VC $PROJECT/scripts/revision/rmod/build_uf_eval_dataset.py \
    --prompts_jsonl $PROJECT/data/gemma2_ufb_part2_test.jsonl \
    --output_dir $RMOD_UF_DATA --chat_template_model google/gemma-2-2b-it
fi
export RMOD_CKPT
if [[ -n ${RMOD_CKPT_OVERRIDE:-} ]]; then
  RMOD_CKPT=$RMOD_CKPT_OVERRIDE
  MODEL_OVERRIDE=(model=gemma_2_2b_multi_head_5)
else
  RMOD_CKPT=$(find "$CACHE/models--Robust-Decoding--uf_5objs_safety_multihead/snapshots" \
    -name last.ckpt -print -quit)
  MODEL_OVERRIDE=()
fi

cd $REPO
CUDA_VISIBLE_DEVICES=$GPU $VR eval.py +experiment=rmod_uf_radar \
  "${MODEL_OVERRIDE[@]}" \
  ++dataset._target_=scripts.revision.rmod.all_multiobjective_dataset.AllMultiObjectiveDataset \
  ++dataset.data_path=$RMOD_UF_DATA ++trainer.devices=[0] \
  ++decoder.tree_depth=$BLOCK ++decoder.num_branches=$K \
  ++decoder.lambda_coef=$LAMBDA ++experiment_name=rmod_${TAG} \
  2>&1 | tee $WORK/gens/logs/rmod_${TAG}.log

PT=$(find $REPO/eval_outputs -name "eval_outputs_rmod_${TAG}_*.pt" -printf '%T@ %p\n' \
  | sort -nr | head -1 | cut -d' ' -f2-)
$VC - "$PT" "$WORK/gens/rmod_${TAG}" <<'PY'
import json, os, sys, torch

def raw_prompt(prompt):
    text = prompt
    start = "<start_of_turn>user\n" if "<start_of_turn>user\n" in text else "user\n"
    end = ("<end_of_turn>\n<start_of_turn>model\n"
           if "<end_of_turn>\n<start_of_turn>model\n" in text else "\nmodel\n")
    if start in text and end in text:
        text = text.split(start, 1)[1].rsplit(end, 1)[0]
    return text.strip()

outputs = torch.load(sys.argv[1], weights_only=False)
outdir = sys.argv[2]
os.makedirs(outdir, exist_ok=True)
records = {}
for row in outputs:
    records.setdefault(raw_prompt(row["prompt"]), []).append(row["response"])
with open(os.path.join(outdir, "generated.json"), "w", encoding="utf-8") as handle:
    json.dump([{"prompt": p, "all_generated_responses": r} for p, r in records.items()], handle)
print("wrote", len(records), "chat-corrected prompts")
PY

CUDA_VISIBLE_DEVICES=$GPU $VC -m on_policy_data_gen.rm_armo_multihead \
  --input_file $WORK/gens/rmod_${TAG}/generated.json --output_dir $WORK/scored \
  --split rmod_${TAG} --indices 6,7,8,9,10 \
  --names instruction_following,truthfulness,honesty,helpfulness,safety \
  --cache_dir $CACHE --batch_size 8 --sample_batch_size 32 \
  2>&1 | tee -a $WORK/gens/logs/rmod_${TAG}.log
echo "[rmod-radar-chat] $TAG done at $(date -Is)"
