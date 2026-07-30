#!/usr/bin/env bash
# Self-contained BPO experiment on polymer 4 GPUs:
#   build 4-arm pairs -> precompute -> train unif/nbs/ks/maxmin -> decode -> judge vs base -> surplus.
# Optional one-judge label noise (covariance experiment): NOISE_OBJ NOISE_LAMBDA.
# Judge at eval is ALWAYS the clean Qwen3-32B (noise only perturbs training targets).
# Usage: bpo_run.sh EXPDIR PARENT GENS_DIR VERDICTS [NOISE_OBJ NOISE_LAMBDA]
set -euo pipefail
EXP=$1; PARENT=$2; GENS=$3; VERD=$4; NOBJ=${5:-}; NLAM=${6:-1.0}; OBJ=${7:-helpfulness,harmlessness}; ETA=${8:-0.0075}; PAIRMODE=${9:-selfplay}
P=/NHNHOME/AIPR/sjkim/MNPO_rev_20260710
V=/NHNHOME/AIPR/sjkim/venv_clean
BASE=/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1/base_objective_screen/hf_ipv4/llama31
JUDGE=/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1/flagship_20260712/cache/huggingface/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
P4=$P/results/p4_8b_saferlhf_table4_20260717
MAN=$P4/dataset_manifest/train_conflict.jsonl
DEC=$P/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py
export PYTHONPATH=$P HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1
mkdir -p $EXP
tag=$(basename $EXP)
NOISEARGS=""; [ -n "$NOBJ" ] && NOISEARGS="--noise-obj $NOBJ --noise-lambda $NLAM"
PVBARGS=""; [ "$PAIRMODE" = "pvb" ] && PVBARGS="--pair-mode pvb --base-files 44=$P4/train_pool/generations/seed44/output_44.json"

echo "[$(basename $EXP)] BUILD ($PAIRMODE)"
$V/bin/python $P/scripts/bpo/build_bpo_pairs.py --verdicts $VERD \
  --policy-files 42=$GENS/seed42/output_42.json 43=$GENS/seed43/output_43.json \
  --out-dir $EXP/pool --split-salt bpo-$tag $NOISEARGS $PVBARGS > $EXP/build.log 2>&1

echo "[$(basename $EXP)] PRECOMPUTE"
CUDA_VISIBLE_DEVICES=0 $V/bin/python -m accelerate.commands.launch \
  --config_file $P/accelerate_configs/single_gpu.yaml --num_processes=1 --main_process_port=31331 \
  -m mnpo_scripts.precompute --model_name_or_path $BASE --ref_model $BASE --history_paths $PARENT \
  --train_dir $EXP/pool/pairs_train.jsonl --test_dir $EXP/pool/pairs_test.jsonl \
  --output_dir $EXP/pool/logps --per_device_train_batch_size 2 --per_device_eval_batch_size 2 \
  --max_length 2048 --max_prompt_length 1024 --apply_chat_template true \
  --auto_insert_empty_system_msg false --ronpo_target_mode none --report_to none \
  > $EXP/precompute.log 2>&1

echo "[$(basename $EXP)] TRAIN 4 arms"
source /NHNHOME/AIPR/sjkim/.secrets/wandb.env
export WANDB_ENTITY=promotion-kim WANDB_PROJECT=mnpo
g=0
for arm in unif nbs ks maxmin; do
  OUT=$EXP/train/$arm/full; mkdir -p $OUT
  if [ -f $OUT/model.safetensors ]; then echo "skip train $arm (exists)"; g=$((g+1)); continue; fi
  cat > $OUT/config.yaml <<EOF
model_name_or_path: $PARENT
torch_dtype: null
attn_implementation: sdpa
dataset_mixer:
  ? $EXP/pool/logps
  : 1.0
dataset_splits:
- train
- test
preprocessing_num_workers: 4
bf16: true
loss_type: ht_mnpo
eta: $ETA
ht_target_column: bpo_target_$arm
ht_target_scale: 1.0
max_history_t: 1
history_weights:
- 1.0
beta: 10.0
reference_anchor_weight: 0.05
preference_sft_weight: 0.005
learning_rate: 5.0e-07
lr_scheduler_type: cosine
warmup_ratio: 0.1
optim: adamw_torch
weight_decay: 0.0
max_grad_norm: 1.0
seed: 42
gradient_accumulation_steps: 16
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
num_train_epochs: 100
max_steps: 900
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
max_length: 2048
max_prompt_length: 1024
do_eval: false
eval_strategy: 'no'
logging_steps: 10
log_level: info
save_strategy: 'no'
save_only_model: true
save_safetensors: true
push_to_hub: false
report_to:
- wandb
output_dir: $OUT
run_name: bpo-$tag-$arm
EOF
  CUDA_VISIBLE_DEVICES=$g $V/bin/python -m accelerate.commands.launch \
    --config_file $P/accelerate_configs/single_gpu.yaml --num_processes=1 --main_process_port=$((31400+g)) \
    -m mnpo_scripts.run_mnpo $OUT/config.yaml > $OUT/train.log 2>&1 &
  g=$((g+1))
done
wait
for arm in unif nbs ks maxmin; do [ -f $EXP/train/$arm/full/model.safetensors ] || { echo "TRAIN_FAILED $arm"; exit 1; }; done
echo "[$(basename $EXP)] TRAIN_DONE"

decode_one(){ local arm=$1 s=$2 gp=$3; local out=$EXP/eval/$arm/gens/seed$s/output_$s.json
  mkdir -p $(dirname $out); [ -s $out ] && return
  CUDA_VISIBLE_DEVICES=$gp $V/bin/python $DEC --manifest $MAN --model $EXP/train/$arm/full \
    --policy-name ev_${arm}_$s --probe none --output $out --seed $s \
    --temperature 0.7 --top-p 0.9 --max-new-tokens 512 --max-model-len 8192 --gpu-memory-utilization 0.88 \
    > $EXP/eval/$arm/decode_$s.log 2>&1; }
echo "[$(basename $EXP)] DECODE"
decode_one unif 42 0 & decode_one unif 43 1 & decode_one nbs 42 2 & decode_one nbs 43 3 & wait
decode_one ks 42 0 & decode_one ks 43 1 & decode_one maxmin 42 2 & decode_one maxmin 43 3 & wait
echo "[$(basename $EXP)] DECODE_DONE"

echo "[$(basename $EXP)] JUDGE+SURPLUS"
for arm in unif nbs ks maxmin; do
  [ -s $EXP/eval/$arm/surplus.json ] && { echo "skip judge $arm (surplus exists)"; continue; }
  OBJ=$OBJ bash $P/scripts/bpo/judge_gen.sh $EXP/eval/$arm/gens $EXP/eval/$arm/verdicts 0 4
  sleep 8; while tmux ls 2>/dev/null | grep -q '^jg'; do sleep 15; done
  cat $EXP/eval/$arm/verdicts/shard*.jsonl > $EXP/eval/$arm/verdicts.jsonl
  $V/bin/python $P/scripts/bpo/eval_bpo_surplus.py --verdicts $EXP/eval/$arm/verdicts.jsonl \
    --label $arm --out $EXP/eval/$arm/surplus.json
done
echo "ALL_DONE $(basename $EXP)"
