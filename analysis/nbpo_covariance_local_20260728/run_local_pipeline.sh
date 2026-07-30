#!/usr/bin/env bash
set -euo pipefail

P=/home/sjkim/MNPO
R=/ext_hdd2/sjkim/nbpo_covariance_local_20260728
S=$P/results/nbpo_covariance_local_20260728
BASE=$R/models/zephyr-7b-sft-full
JUDGE=$R/models/Qwen3-32B
TRAIN=$S/data/train.jsonl
TEST=$S/data/test.jsonl
TV=/home/sjkim/anaconda3/envs/vllm/bin/python
TP=/home/sjkim/anaconda3/envs/ronpo-rev/bin/python
DEC=$P/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py
export PYTHONPATH=$P
export HF_HOME=$R/hf_home HF_XET_CACHE=$R/hf_xet
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_ENTITY=promotion-kim WANDB_PROJECT=mnpo
mkdir -p "$R"/{gens,verdicts,pairs_clean,pairs_noise03,pre,train,eval,logs}

need_base() {
  [[ -s $BASE/model.safetensors.index.json ]] &&
  [[ $(find "$BASE" -maxdepth 1 -name '*.safetensors' | wc -l) -eq 3 ]]
}
need_judge() {
  [[ -s $JUDGE/model.safetensors.index.json ]] &&
  [[ $(find "$JUDGE" -maxdepth 1 -name 'model-*.safetensors' | wc -l) -eq 17 ]]
}
until need_base; do
  echo "$(date --iso-8601=seconds) waiting_for_base" >> "$R/logs/pipeline.log"
  sleep 60
done

decode() {
  local manifest=$1 model=$2 name=$3 seed=$4 gpu=$5 temp=$6 top_p=$7 out=$8
  [[ -s $out ]] && return
  mkdir -p "$(dirname "$out")"
  CUDA_VISIBLE_DEVICES=$gpu "$TV" "$DEC" --manifest "$manifest" --model "$model" \
    --policy-name "$name" --probe none --output "$out" --seed "$seed" \
    --temperature "$temp" --top-p "$top_p" --max-new-tokens 512 \
    --max-model-len 2048 --gpu-memory-utilization 0.86 \
    > "${out%.json}.log" 2>&1
}

decode "$TRAIN" "$BASE" base42 42 0 0.9 0.95 "$R/gens/g42.json" &
decode "$TRAIN" "$BASE" base43 43 1 0.9 0.95 "$R/gens/g43.json" &
decode "$TRAIN" "$BASE" base44 44 2 0.9 0.95 "$R/gens/g44.json" &
wait
decode "$TRAIN" "$BASE" base45 45 0 0.9 0.95 "$R/gens/g45.json" &
decode "$TRAIN" "$BASE" base46 46 1 0.9 0.95 "$R/gens/g46.json" &
decode "$TEST" "$BASE" eval_base46 46 2 0.7 0.9 "$R/eval/base46.json" &
wait

until need_judge; do
  echo "$(date --iso-8601=seconds) waiting_for_judge" >> "$R/logs/pipeline.log"
  sleep 60
done
for gpu in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$gpu "$TV" "$P/scripts/bpo/judge_bpo.py" \
    --policy-files 42="$R/gens/g42.json" 43="$R/gens/g43.json" \
      44="$R/gens/g44.json" 45="$R/gens/g45.json" \
    --reference-files 46="$R/gens/g46.json" \
    --objectives helpfulness,harmlessness,honesty --judge-model-path "$JUDGE" \
    --max-model-len 4096 --output "$R/verdicts/shard$gpu.jsonl" \
    --num-shards 3 --shard-index "$gpu" > "$R/verdicts/judge$gpu.log" 2>&1 &
done
wait
cat "$R"/verdicts/shard*.jsonl > "$R/verdicts/all.jsonl"

"$TP" "$P/scripts/bpo/build_bpo_pairs.py" \
  --verdicts "$R/verdicts/all.jsonl" \
  --policy-files 42="$R/gens/g42.json" 43="$R/gens/g43.json" \
    44="$R/gens/g44.json" 45="$R/gens/g45.json" \
  --out-dir "$R/pairs_clean" --test-prompts 32 --split-salt nbpo-cov-local \
  --all-seed-pairs > "$R/logs/build_clean.log" 2>&1
"$TP" "$P/analysis/nbpo_covariance_local_20260728/add_noisy_targets.py" \
  --verdicts "$R/verdicts/all.jsonl" --pairs-dir "$R/pairs_clean" \
  --out-dir "$R/pairs_noise03" > "$R/logs/build_noise.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "$TP" -m accelerate.commands.launch \
  --config_file "$P/accelerate_configs/single_gpu.yaml" --num_processes=1 \
  --main_process_port=32600 -m mnpo_scripts.precompute \
  --model_name_or_path "$BASE" --ref_model "$BASE" --history_paths "$BASE" \
  --train_dir "$R/pairs_noise03/pairs_train.jsonl" \
  --test_dir "$R/pairs_noise03/pairs_test.jsonl" --output_dir "$R/pre" \
  --per_device_train_batch_size 2 --per_device_eval_batch_size 2 \
  --max_length 1536 --max_prompt_length 1024 --apply_chat_template true \
  --auto_insert_empty_system_msg false --ronpo_target_mode none --report_to none \
  > "$R/logs/precompute.log" 2>&1

"$TP" "$P/analysis/nbpo_covariance_local_20260728/make_train_configs.py" \
  --root "$R" --base "$BASE" --dataset "$R/pre"
train_queue() {
  local gpu=$1; shift
  for name in "$@"; do
    [[ -s $R/train/$name/model.safetensors ]] && continue
    CUDA_VISIBLE_DEVICES=$gpu "$TP" -m accelerate.commands.launch \
      --config_file "$P/accelerate_configs/single_gpu.yaml" --num_processes=1 \
      --main_process_port=$((32700+gpu)) -m mnpo_scripts.run_mnpo \
      "$R/train/$name/config.yaml" > "$R/train/$name/train.log" 2>&1
  done
}
train_queue 0 clean_nbs clean_maxmin noise03_unif &
train_queue 1 clean_ks noise03_nbs noise03_maxmin &
train_queue 2 clean_unif noise03_ks &
wait

names=(clean_nbs clean_ks clean_unif clean_maxmin noise03_nbs noise03_ks noise03_unif noise03_maxmin)
for i in "${!names[@]}"; do
  name=${names[$i]}; gpu=$((i%3))
  decode "$TEST" "$R/train/$name" "$name" 42 "$gpu" 0.7 0.9 \
    "$R/eval/$name/gens.json" &
  (( i%3 == 2 )) && wait
done
wait
for name in "${names[@]}"; do
  "$TP" "$P/scripts/revision/flagship/stability_gate_corrected.py" \
    --base "$R/eval/base46.json" --candidate "$R/eval/$name/gens.json" \
    --output "$R/eval/$name/stability.json" --expected-records 256
done

judge_eval() {
  local name=$1 gpu=$2
  mkdir -p "$R/eval/$name/verdicts"
  CUDA_VISIBLE_DEVICES=$gpu "$TV" "$P/scripts/bpo/judge_bpo.py" \
    --policy-files 42="$R/eval/$name/gens.json" \
    --reference-files 46="$R/eval/base46.json" \
    --objectives helpfulness,harmlessness,honesty --judge-model-path "$JUDGE" \
    --max-model-len 4096 --output "$R/eval/$name/verdicts/shard.jsonl" \
    > "$R/eval/$name/judge.log" 2>&1
  cp "$R/eval/$name/verdicts/shard.jsonl" "$R/eval/$name/verdicts.jsonl"
}
for ((i=0; i<${#names[@]}; i+=3)); do
  for j in 0 1 2; do
    (( i+j < ${#names[@]} )) && judge_eval "${names[$((i+j))]}" "$j" &
  done
  wait
done

"$TP" "$P/analysis/nbpo_covariance_local_20260728/aggregate_covariance.py" \
  --root "$R" --out "$S/paired_summary.json"
sha256sum "$R"/eval/*/verdicts.jsonl "$R"/eval/*/gens.json \
  > "$S/artifacts.sha256"
date --iso-8601=seconds > "$S/COMPLETE"
