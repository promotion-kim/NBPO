#!/usr/bin/env bash
# Assemble + run RMOD blockwise robust decoding for the Task-1 radar, then
# score the generations with the shared ArmoRM 5-head scorer.
#   GPU=0 LAMBDA=0.5 bash run_rmod_radar.sh
set -euo pipefail
SJ=/NHNHOME/AIPR/sjkim
REPO=$SJ/rmod_20260720/repo
PROJECT=$SJ/MNPO_rev_20260720
WORK=$SJ/rmod_20260720/radar
CACHE=$SJ/baseline_repair_1p5b_20260714/cache/huggingface/hub
VR=$SJ/rmod_20260720/venv_rmod/bin/python
VC=$SJ/venv_clean/bin/python
GPU="${GPU:-0}"; LAMBDA="${LAMBDA:-0.5}"
TAG="${TAG:-l${LAMBDA//./p}}"
EXTRA="${EXTRA:-}"
export HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface
export HF_HUB_CACHE=$CACHE HUGGINGFACE_HUB_CACHE=$CACHE
export HUGGINGFACE_HUB_TOKEN=${HF_TOKEN:?}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PROJECT
mkdir -p $WORK/gens/logs

# 1. UF eval dataset (once)
export RMOD_UF_DATA=$WORK/uf_eval_ds
if [[ ! -d $RMOD_UF_DATA ]]; then
  $VC $PROJECT/scripts/revision/rmod/build_uf_eval_dataset.py \
    --prompts_jsonl $PROJECT/data/gemma2_ufb_part2_test.jsonl --output_dir $RMOD_UF_DATA
fi

# 2. RMOD generation (no oracle; scored separately)
export RMOD_CKPT=$(ls -d $CACHE/models--Robust-Decoding--uf_5objs_safety_multihead/snapshots/*/last.ckpt | head -1)
mkdir -p $REPO/eval_outputs
cd $REPO
CUDA_VISIBLE_DEVICES=$GPU $VR eval.py +experiment=rmod_uf_radar \
  ++trainer.devices=[0] ++decoder.lambda_coef=$LAMBDA \
  ++experiment_name=rmod_${TAG} $EXTRA \
  2>&1 | tee $WORK/gens/logs/rmod_${TAG}.log

# 3. extract responses -> jsonl, score with ArmoRM 5 heads
PT=$(ls -t $REPO/eval_outputs/eval_outputs_rmod_${TAG}_*.pt | head -1)
$VC - "$PT" "$WORK/gens/rmod_${TAG}" <<'PY'
import sys, os, json, torch
pt, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
outs = torch.load(pt, weights_only=False)
recs = {}
for o in outs:
    p = o["prompt"]; recs.setdefault(p, []).append(o["response"])
with open(os.path.join(outdir, "generated.json"), "w") as f:
    json.dump([{"prompt": p, "all_generated_responses": r} for p, r in recs.items()], f)
print("wrote", len(recs), "prompts")
PY
CUDA_VISIBLE_DEVICES=$GPU $VC -m on_policy_data_gen.rm_armo_multihead \
  --input_file $WORK/gens/rmod_${TAG}/generated.json --output_dir $WORK/scored --split rmod_${TAG} \
  --indices 6,7,8,9,10 --names instruction_following,truthfulness,honesty,helpfulness,safety \
  --cache_dir $CACHE --batch_size 8 --sample_batch_size 32 \
  2>&1 | tee -a $WORK/gens/logs/rmod_${TAG}.log
echo "[rmod-radar] $TAG done at $(date -Is)"
