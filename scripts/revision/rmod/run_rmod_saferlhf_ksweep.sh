#!/usr/bin/env bash
# RMOD blockwise robust decoding on the SafeRLHF panel, sweeping K = num_branches
# (Fig-3 orange trajectory). Generates from Llama-3.1-8B base steered by the
# token-matched 2-head VF, then Beaver-scores each K. One K per GPU via env.
#   GPU=2 K=16 bash run_rmod_saferlhf_ksweep.sh
set -uo pipefail
SJ=/NHNHOME/AIPR/sjkim
REPO=$SJ/rmod_20260720/repo
P10=$SJ/MNPO_rev_20260710
CACHE=$SJ/baseline_repair_1p5b_20260714/cache/huggingface/hub
VR=$SJ/rmod_20260720/venv_rmod/bin/python
VC=$SJ/venv_clean/bin/python
GPU="${GPU:-0}"; K="${K:-16}"; NPROMPTS="${NPROMPTS:-250}"; TREEDEPTH="${TREEDEPTH:-8}"
export HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface HF_HUB_CACHE=$CACHE HUGGINGFACE_HUB_CACHE=$CACHE
export HUGGINGFACE_HUB_TOKEN=${HF_TOKEN:?} TOKENIZERS_PARALLELISM=false
export RMOD_SAFERLHF_CKPT="${CKPT:-$(ls $REPO/outputs/2026-07-21/15-18-16/saferlhf_vf/*/*/transformer-step*.ckpt | head -1)}"
OUT="${OUTDIR:-$SJ/rmod_20260720/saferlhf_ksweep}"; mkdir -p $OUT/logs
REWARD=$SJ/revision_qwen3_8b/full_iter1/flagship_20260712/cache/huggingface/models--PKU-Alignment--beaver-7b-v1.0-reward/snapshots/375cd6a9f0d7e339d2199b05ba129a4a8906596d
COST=$SJ/revision_qwen3_8b/full_iter1/flagship_20260712/cache/huggingface/models--PKU-Alignment--beaver-7b-v1.0-cost/snapshots/c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28

# 1. eval dataset (subset of the frozen panel), 2 placeholder objs
export RMOD_SAFERLHF_DATA=$OUT/eval_ds_${NPROMPTS}
if [[ ! -d $RMOD_SAFERLHF_DATA ]]; then
  $VC - "$P10/results/p8_stage4_fresh_default_test_20260718/dataset_manifest/fresh_default_test_1000.jsonl" \
        "$RMOD_SAFERLHF_DATA" "$NPROMPTS" <<'PY'
import sys, json
from datasets import Dataset
src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
prompts, seen = [], set()
for line in open(src, encoding="utf-8"):
    p = json.loads(line)["prompt"]
    if p and p not in seen:
        seen.add(p); prompts.append(p)
    if len(prompts) >= n:
        break
rows = {"prompt": prompts, "response": ["placeholder response text unused"] * len(prompts),
        "obj0": [0.0] * len(prompts), "obj1": [0.0] * len(prompts)}
Dataset.from_dict(rows).save_to_disk(out)
print("[saferlhf-eval-ds]", len(prompts), "prompts ->", out)
PY
fi

# 2. RMOD generation at K = num_branches
cd $REPO
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$REPO $VR eval.py +experiment=rmod_saferlhf_ksweep \
  ++trainer.devices=[0] ++decoder.num_branches=$K ++decoder.tree_depth=$TREEDEPTH \
  ++experiment_name=rmod_saferlhf_k${K} \
  2>&1 | tee $OUT/logs/rmod_k${K}.log

# 3. extract responses -> fmt.jsonl
PT=$(ls -t $REPO/eval_outputs/eval_outputs_rmod_saferlhf_k${K}_*.pt | head -1)
$VC - "$PT" "$OUT/k${K}_fmt.jsonl" <<'PY'
import sys, json, torch
pt, out = sys.argv[1], sys.argv[2]
outs = torch.load(pt, weights_only=False)
seen = {}
for o in outs:
    seen.setdefault(o["prompt"], o["response"])
with open(out, "w") as f:
    for p, r in seen.items():
        f.write(json.dumps({"prompt": p, "all_generated_responses": [r]}) + "\n")
print("extracted", len(seen), "prompts ->", out)
PY

# 4. Beaver reward + cost
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$P10 $VC $P10/on_policy_data_gen/rm_beaver_reward.py \
  --model_name $REWARD --model_revision 375cd6a9f0d7e339d2199b05ba129a4a8906596d \
  --input_file $OUT/k${K}_fmt.jsonl --output_file $OUT/k${K}_helpfulness.jsonl --batch_size 8 --sample_batch_size 8 \
  > $OUT/logs/rw_k${K}.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$P10 $VC $P10/on_policy_data_gen/rm_beaver_cost.py \
  --model_name $COST --model_revision c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28 \
  --input_file $OUT/k${K}_fmt.jsonl --output_file $OUT/k${K}_harmlessness.jsonl --batch_size 8 --sample_batch_size 8 \
  > $OUT/logs/ct_k${K}.log 2>&1
$VC - "$OUT" "$K" <<'PY'
import sys, json, statistics as st
d, k = sys.argv[1], sys.argv[2]
h = [json.loads(l)["all_rm_scores"][0] for l in open(f"{d}/k{k}_helpfulness.jsonl")]
c = [json.loads(l)["all_rm_scores"][0] for l in open(f"{d}/k{k}_harmlessness.jsonl")]
rec = {"k": int(k), "helpful": round(st.mean(h), 4), "harmless": round(st.mean(c), 4), "n": len(h)}
open(f"{d}/k{k}_summary.json", "w").write(json.dumps(rec))
print("[rmod-ksweep]", rec)
PY
echo "[rmod-saferlhf K=$K] done $(date -Is)"
