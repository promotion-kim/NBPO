#!/usr/bin/env bash
# Chat-template-correct RMOD K-sweep on the same 1,000-prompt SafeRLHF panel
# used by the RONPO stage trajectory. One K per GPU.
set -euo pipefail
SJ=/NHNHOME/AIPR/sjkim
REPO=$SJ/rmod_20260720/repo
PROJECT=$SJ/MNPO_rev_20260720
P10=$SJ/MNPO_rev_20260710
CACHE=$SJ/baseline_repair_1p5b_20260714/cache/huggingface/hub
VR=$SJ/rmod_20260720/venv_rmod/bin/python
VC=$SJ/venv_clean/bin/python
GPU=${GPU:-0}; K=${K:-16}; BLOCK=${BLOCK:-16}; NPROMPTS=${NPROMPTS:-1000}
SHARD_COUNT=${SHARD_COUNT:-1}; SHARD_INDEX=${SHARD_INDEX:-0}
SHARD_SUFFIX=""
if [[ $SHARD_COUNT -gt 1 ]]; then
  SHARD_SUFFIX=_shard${SHARD_INDEX}of${SHARD_COUNT}
fi
SRC=$P10/results/p8_stage4_fresh_default_test_20260718/dataset_manifest/fresh_default_test_1000.jsonl
LLAMA=$SJ/revision_qwen3_8b/full_iter1/base_objective_screen/hf_ipv4/llama31
OUT=$SJ/rmod_20260720/saferlhf_ksweep_chat_b${BLOCK}_n${NPROMPTS}
export HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface
export HF_HUB_CACHE=$CACHE HUGGINGFACE_HUB_CACHE=$CACHE
export TOKENIZERS_PARALLELISM=false PYTHONPATH=$PROJECT:$REPO
if [[ -z ${HUGGINGFACE_HUB_TOKEN:-} ]]; then
  export HUGGINGFACE_HUB_TOKEN
  HUGGINGFACE_HUB_TOKEN=$(sed -n 's/.*HUGGINGFACE_HUB_TOKEN=\([^ ]*\).*/\1/p' \
    $PROJECT/scripts/revision/rmod/run_rmod_saferlhf_ksweep.sh | head -1)
fi
export RMOD_SAFERLHF_DATA=$OUT/eval_ds
CKPT_LIST=$SJ/rmod_20260720/saferlhf_vf_disjoint_run/checkpoints.txt
while [[ ! -s $CKPT_LIST ]]; do
  sleep 30
done
export RMOD_SAFERLHF_CKPT
RMOD_SAFERLHF_CKPT=$(head -1 "$CKPT_LIST")
[[ -s $RMOD_SAFERLHF_CKPT ]]
mkdir -p $OUT/logs

if [[ ! -d $RMOD_SAFERLHF_DATA ]]; then
  $VC $PROJECT/scripts/revision/rmod/build_uf_eval_dataset.py \
    --prompts_jsonl $SRC --output_dir $RMOD_SAFERLHF_DATA \
    --chat_template_model $LLAMA
fi

cd $REPO
CUDA_VISIBLE_DEVICES=$GPU $VR eval.py +experiment=rmod_saferlhf_ksweep \
  ++dataset._target_=scripts.revision.rmod.all_multiobjective_dataset.AllMultiObjectiveDataset \
  ++dataset.num_shards=$SHARD_COUNT ++dataset.shard_index=$SHARD_INDEX \
  ++dataset.data_path=$RMOD_SAFERLHF_DATA ++eval_subset=$NPROMPTS \
  ++trainer.devices=[0] ++decoder.num_branches=$K ++decoder.tree_depth=$BLOCK \
  ++experiment_name=rmod_saferlhf_chat_b${BLOCK}_k${K}_n${NPROMPTS}${SHARD_SUFFIX} \
  2>&1 | tee $OUT/logs/rmod_k${K}${SHARD_SUFFIX}.log

TAG=rmod_saferlhf_chat_b${BLOCK}_k${K}_n${NPROMPTS}${SHARD_SUFFIX}
PT=$(find $REPO/eval_outputs -name "eval_outputs_${TAG}_*.pt" -printf '%T@ %p\n' \
  | sort -nr | head -1 | cut -d' ' -f2-)
$VC - "$PT" "$SRC" "$OUT/k${K}${SHARD_SUFFIX}_fmt.jsonl" "$NPROMPTS" <<'PY'
import json, sys, torch
pt, source, out, limit = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
raw, seen = [], set()
for line in open(source, encoding="utf-8"):
    prompt = json.loads(line)["prompt"]
    if prompt and prompt not in seen:
        seen.add(prompt); raw.append(prompt)
    if len(raw) >= limit:
        break
outputs = torch.load(pt, weights_only=False)
records = {}
for row in outputs:
    rendered = row["prompt"]
    matches = [prompt for prompt in raw if prompt in rendered]
    prompt = max(matches, key=len) if matches else rendered
    records.setdefault(prompt, row["response"])
with open(out, "w", encoding="utf-8") as handle:
    for prompt, response in records.items():
        handle.write(json.dumps({"prompt": prompt, "all_generated_responses": [response]}) + "\n")
print("extracted", len(records), "chat-corrected prompts")
PY

REWARD=$SJ/revision_qwen3_8b/full_iter1/flagship_20260712/cache/huggingface/models--PKU-Alignment--beaver-7b-v1.0-reward/snapshots/375cd6a9f0d7e339d2199b05ba129a4a8906596d
COST=$SJ/revision_qwen3_8b/full_iter1/flagship_20260712/cache/huggingface/models--PKU-Alignment--beaver-7b-v1.0-cost/snapshots/c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$P10 $VC $P10/on_policy_data_gen/rm_beaver_reward.py \
  --model_name $REWARD --model_revision 375cd6a9f0d7e339d2199b05ba129a4a8906596d \
  --input_file $OUT/k${K}${SHARD_SUFFIX}_fmt.jsonl --output_file $OUT/k${K}${SHARD_SUFFIX}_helpfulness.jsonl \
  --batch_size 8 --sample_batch_size 8 > $OUT/logs/reward_k${K}${SHARD_SUFFIX}.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$P10 $VC $P10/on_policy_data_gen/rm_beaver_cost.py \
  --model_name $COST --model_revision c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28 \
  --input_file $OUT/k${K}${SHARD_SUFFIX}_fmt.jsonl --output_file $OUT/k${K}${SHARD_SUFFIX}_harmlessness.jsonl \
  --batch_size 8 --sample_batch_size 8 > $OUT/logs/cost_k${K}${SHARD_SUFFIX}.log 2>&1
$VC - "$OUT" "$K" "$BLOCK" "$SHARD_SUFFIX" "$SHARD_INDEX" "$SHARD_COUNT" "$RMOD_SAFERLHF_CKPT" <<'PY'
import json, statistics as st, sys
root, k, block, suffix = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
shard_index, shard_count, checkpoint = int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
helpful = [json.loads(x)["all_rm_scores"][0] for x in open(f"{root}/k{k}{suffix}_helpfulness.jsonl")]
harmless = [json.loads(x)["all_rm_scores"][0] for x in open(f"{root}/k{k}{suffix}_harmlessness.jsonl")]
record = {"k": int(k), "block_size": block, "helpful": st.mean(helpful),
          "harmless": st.mean(harmless), "n": len(helpful), "chat_template": True,
          "shard_index": shard_index, "shard_count": shard_count,
          "value_function_checkpoint": checkpoint}
open(f"{root}/k{k}{suffix}_summary.json", "w").write(json.dumps(record, indent=2) + "\n")
print(record)
PY
echo "[rmod-saferlhf-chat K=$K] done $(date -Is)"
