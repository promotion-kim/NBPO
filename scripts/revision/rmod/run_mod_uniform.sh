#!/usr/bin/env bash
# UNIFORM/MOD decoding baseline for the radar. Generation only, then score with
# the shared ArmoRM 5-head scorer.  GPU=0 bash run_mod_uniform.sh
set -euo pipefail
SJ=/NHNHOME/AIPR/sjkim
REPO=$SJ/rmod_20260720/repo; PROJECT=$SJ/MNPO_rev_20260720; WORK=$SJ/rmod_20260720/radar
CACHE=$SJ/baseline_repair_1p5b_20260714/cache/huggingface/hub
VR=$SJ/rmod_20260720/venv_rmod/bin/python; VC=$SJ/venv_clean/bin/python
GPU="${GPU:-0}"; TAG=mod_uniform
export HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface HF_HUB_CACHE=$CACHE HUGGINGFACE_HUB_CACHE=$CACHE
export HUGGINGFACE_HUB_TOKEN=${HF_TOKEN:?} TOKENIZERS_PARALLELISM=false PYTHONPATH=$PROJECT
export RMOD_UF_DATA=$WORK/uf_eval_ds RMOD_CKPT=$(ls -d $CACHE/models--Robust-Decoding--uf_5objs_safety_multihead/snapshots/*/last.ckpt|head -1)
mkdir -p $REPO/eval_outputs $WORK/gens/logs
cd $REPO
CUDA_VISIBLE_DEVICES=$GPU $VR eval.py +experiment=mod_uniform_uf ++trainer.devices=[0] \
  ++experiment_name=$TAG 2>&1 | tee $WORK/gens/logs/${TAG}.log
PT=$(ls -t $REPO/eval_outputs/eval_outputs_${TAG}_*.pt|head -1)
$VC - "$PT" "$WORK/gens/$TAG" <<'PY'
import sys, os, json, torch
pt, outdir = sys.argv[1], sys.argv[2]; os.makedirs(outdir, exist_ok=True)
outs = torch.load(pt, weights_only=False); recs = {}
for o in outs: recs.setdefault(o["prompt"], []).append(o["response"])
json.dump([{"prompt": p, "all_generated_responses": r} for p, r in recs.items()], open(os.path.join(outdir,"generated.json"),"w"))
print("wrote", len(recs), "prompts")
PY
CUDA_VISIBLE_DEVICES=$GPU $VC -m on_policy_data_gen.rm_armo_multihead \
  --input_file $WORK/gens/$TAG/generated.json --output_dir $WORK/scored --split $TAG \
  --indices 6,7,8,9,10 --names instruction_following,truthfulness,honesty,helpfulness,safety \
  --cache_dir $CACHE --batch_size 16 --sample_batch_size 32 2>&1 | tail -2
echo "[mod-uniform] done at $(date -Is)"
