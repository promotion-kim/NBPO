#!/usr/bin/env bash
set -euo pipefail
SJ=/NHNHOME/AIPR/sjkim
P=$SJ/MNPO_rev_20260720
ROOT=$SJ/uf5_inpo_warmstart_robustft_20260724
PY=$SJ/venv_clean/bin/python
CACHE=$SJ/baseline_repair_1p5b_20260714/cache/huggingface/hub
PARENT=$SJ/ronpo_gemma_baselines_s1/inpo
BASE_POOL=$SJ/ronpo_gemma_20260720/pool
NAMES=(instruction_following truthfulness honesty helpfulness safety)
export PYTHONPATH=$P HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface
export HF_HUB_CACHE=$CACHE HUGGINGFACE_HUB_CACHE=$CACHE
export TOKENIZERS_PARALLELISM=false MNPO_DISABLE_APEX=1 NCCL_SOCKET_IFNAME=lo
export WANDB_ENTITY=promotion-kim WANDB_PROJECT=mnpo WANDB_MODE=online WANDB_DIR=$ROOT/wandb
mkdir -p "$ROOT"/{logs,wandb,pool,scored,pairs,precomputed,train,eval,hf_uploads}

for seed in 100 101 102 103 104 105; do
  while [[ ! -s "$ROOT/pool/PARENT_SEED_${seed}_COMPLETE" ]]; do sleep 30; done
done

for split in train test; do
  mkdir -p "$ROOT/pool/$split/merged"
  rm -f "$ROOT/pool/$split/merged"/output_*.json "$ROOT/pool/$split/merged/all_outputs.json"
  for seed in 13 21 42 79; do
    cp "$BASE_POOL/$split/merged/output_${seed}.json" "$ROOT/pool/$split/merged/"
  done
  for seed in 100 101 102 103 104 105; do
    cp "$ROOT/pool/$split/parent/output_${seed}.json" "$ROOT/pool/$split/merged/"
  done
  "$PY" -m on_policy_data_gen.post_process \
    --generation_file_dir "$ROOT/pool/$split/merged" \
    > "$ROOT/logs/merge_${split}.log" 2>&1
done
date -Is > "$ROOT/pool/POOL_READY"

if [[ ! -s "$ROOT/scored/READY" ]]; then
  if [[ ! -s "$ROOT/scored/SCORES_0_3_COMPLETE" || ! -s "$ROOT/scored/SCORES_4_5_COMPLETE" ]]; then
    bash "$P/analysis/uf5_inpo_warmstart_robustft_20260724/score_pool_worker.sh" 0 3 &
    score_pid=$!
    while [[ ! -s "$ROOT/scored/SCORES_4_5_COMPLETE" ]]; do sleep 30; done
    wait "$score_pid"
  fi
  for split in train test; do
    expected=$("$PY" - "$ROOT/pool/$split/merged/all_outputs.json" <<'PY'
import json,sys
print(len(json.load(open(sys.argv[1]))))
PY
)
    for name in "${NAMES[@]}"; do
      out=$ROOT/scored/${split}_${name}.jsonl
      : > "$out"
      for shard in 0 1 2 3 4 5; do
        cat "$out.shard${shard}" >> "$out"
      done
      test "$(wc -l < "$out")" -eq "$expected"
    done
  done
  date -Is > "$ROOT/scored/READY"
fi

if [[ ! -s "$ROOT/pairs/READY" ]]; then
  train_scores=()
  test_scores=()
  for name in "${NAMES[@]}"; do
    train_scores+=("$name=$ROOT/scored/train_${name}.jsonl")
    test_scores+=("$name=$ROOT/scored/test_${name}.jsonl")
  done
  "$PY" -m mnpo_scripts.build_multi_objective_dataset \
    --scored_files "${train_scores[@]}" \
    --mnpo_output "$ROOT/pairs/train_avg.jsonl" \
    --ronpo_output "$ROOT/pairs/train_ronpo.jsonl" \
    --normalization minmax --ronpo_pair_strategy sigma \
    --ronpo_policy_pair_mode expected_relative_policy_vs_policy \
    --ronpo_policy_samples_per_atom 1 --adversary_steps 25 \
    --adversary_alpha 1.0 --adversary_kappa 0.05 \
    --preference_scale 8.0 --pairs_per_prompt 2 \
    > "$ROOT/logs/build_train_pairs.log" 2>&1
  "$PY" -m mnpo_scripts.build_multi_objective_dataset \
    --scored_files "${test_scores[@]}" \
    --mnpo_output "$ROOT/pairs/test_avg.jsonl" \
    --ronpo_output "$ROOT/pairs/test_ronpo.jsonl" \
    --normalization minmax --ronpo_pair_strategy sigma \
    --ronpo_policy_pair_mode expected_relative_policy_vs_policy \
    --ronpo_policy_samples_per_atom 1 --adversary_steps 25 \
    --adversary_alpha 1.0 --adversary_kappa 0.05 \
    --preference_scale 8.0 --pairs_per_prompt 2 \
    > "$ROOT/logs/build_test_pairs.log" 2>&1
  date -Is > "$ROOT/pairs/READY"
fi

precompute() {
  local arm=$1 train=$2 test=$3 gpus=$4 port=$5
  local out=$ROOT/precomputed/$arm
  [[ -s "$out/COMPLETE" ]] && return
  rm -rf "$out/raw" "$out/final"
  CUDA_VISIBLE_DEVICES=$gpus "$PY" -m accelerate.commands.launch \
    --num_processes=2 --main_process_port="$port" -m mnpo_scripts.precompute \
    --model_name_or_path "$PARENT" --ref_model "$PARENT" \
    --history_paths "$PARENT" --train_dir "$train" --test_dir "$test" \
    --output_dir "$out/raw" --per_device_train_batch_size 8 \
    --max_length 2048 --max_prompt_length 1800 \
    --apply_chat_template true --auto_insert_empty_system_msg false \
    --ronpo_target_mode none --sanity_check False \
    > "$ROOT/logs/precompute_${arm}.log" 2>&1
  if [[ $arm == ronpo ]]; then
    "$PY" -m mnpo_scripts.build_os_ronpo_targets --input_dir "$out/raw" \
      --output_dir "$out/final" --kappas 0.05 --num_proc 8 \
      > "$ROOT/logs/build_targets_ronpo.log" 2>&1
  else
    ln -sfn raw "$out/final"
  fi
  date -Is > "$out/COMPLETE"
}
precompute inpo "$ROOT/pairs/train_avg.jsonl" "$ROOT/pairs/test_avg.jsonl" 0,1 32310 &
p0=$!
precompute ronpo "$ROOT/pairs/train_ronpo.jsonl" "$ROOT/pairs/test_ronpo.jsonl" 2,3 32311 &
p1=$!
wait "$p0"
wait "$p1"
date -Is > "$ROOT/precomputed/READY"

train_arm() {
  local arm=$1 loss=$2 gpus=$3 port=$4 accum=$5
  [[ -s "$ROOT/train/$arm/all_results.json" ]] && return
  local target=()
  if [[ $loss == ronpo ]]; then
    target=(--ronpo_alpha=1.0 --ronpo_tau=0.05 --ronpo_target_column=target_os_k0p05)
  else
    target=(--eta=0.0075 --ratio=0.3333)
  fi
  CUDA_VISIBLE_DEVICES=$gpus "$PY" -m accelerate.commands.launch \
    --config_file "$P/accelerate_configs/multi_gpu.yaml" --num_processes=2 \
    --main_process_port="$port" -m mnpo_scripts.run_mnpo \
    "$P/training_configs/ronpo/qwen2.5-1.5b-instruct-ronpo-multiobj-iter1.yaml" \
    --model_name_or_path="$PARENT" --dataset_mixer="$ROOT/precomputed/$arm/final:1.0" \
    --loss_type="$loss" --max_history_t=1 --history_weights=1.0 \
    --learning_rate=1.0e-6 --warmup_ratio=0.1 --num_train_epochs=1 \
    --seed=42 --max_steps=1800 --per_device_train_batch_size=1 \
    --gradient_accumulation_steps="$accum" --gradient_checkpointing=true \
    --max_length=2048 --max_prompt_length=1800 --do_eval=false \
    --eval_strategy=no --generate_during_eval=false --save_strategy=steps \
    --save_steps=600 --save_total_limit=1 --logging_steps=10 \
    --output_dir="$ROOT/train/$arm" --run_name="ronpo-uf5-inpo-warmstart-${arm}-s42" \
    "${target[@]}" > "$ROOT/logs/train_${arm}.log" 2>&1
  test -s "$ROOT/train/$arm/all_results.json"
}
train_arm inpo inpo 0,1 32410 8 &
t0=$!
train_arm ronpo ronpo 2,3 32411 8 &
t1=$!
wait "$t0"
wait "$t1"
date -Is > "$ROOT/TRAINING_COMPLETE"

# Fresh panel remains unopened for policy inference until both final checkpoints exist.
for tag in base parent inpo ronpo; do mkdir -p "$ROOT/eval/$tag"; done
decode_eval() {
  local tag=$1 model=$2 gpu=$3
  if [[ -s "$ROOT/eval/$tag/DECODE_COMPLETE" ]]; then
    "$PY" - "$ROOT/eval/$tag/output_42.json" <<'PY'
import json,sys
assert len(json.load(open(sys.argv[1]))) == 646
PY
    return
  fi
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u -m on_policy_data_gen.decode \
    --data_dir "$ROOT/dataset_manifest/fresh_part1_test_646.jsonl" \
    --model "$model" --seeds 42 --output_dir "$ROOT/eval/$tag" --num_gpu 1 \
    --temperature 0.8 --top_p 0.95 --max_tokens 128 --batch_size 256 \
    --dtype bfloat16 --cache_dir "$CACHE" > "$ROOT/logs/decode_fresh_${tag}.log" 2>&1
  "$PY" - "$ROOT/eval/$tag/output_42.json" <<'PY'
import json,sys
assert len(json.load(open(sys.argv[1]))) == 646
PY
  date -Is > "$ROOT/eval/$tag/DECODE_COMPLETE"
}
decode_eval base google/gemma-2-2b-it 0 &
d0=$!
decode_eval parent "$PARENT" 1 &
d1=$!
decode_eval inpo "$ROOT/train/inpo" 2 &
d2=$!
decode_eval ronpo "$ROOT/train/ronpo" 3 &
d3=$!
wait "$d0"
wait "$d1"
wait "$d2"
wait "$d3"

for arm in inpo ronpo; do
  set +e
  "$PY" "$P/scripts/revision/flagship/stability_gate_corrected.py" \
    --base "$ROOT/eval/base/output_42.json" \
    --candidate "$ROOT/eval/$arm/output_42.json" --expected-records 646 \
    --min-length-ratio 0.33 --max-length-ratio 2.0 --max-repeat-run 20 \
    --output "$ROOT/eval/$arm/stability_gate.json" \
    > "$ROOT/logs/stability_${arm}.log" 2>&1
  rc=$?
  set -e
  if (( rc != 0 )); then
    date -Is > "$ROOT/eval/$arm/GATE_FAILED"
  else
    date -Is > "$ROOT/eval/$arm/GATE_PASSED"
    "$PY" "$P/analysis/uf5_inpo_warmstart_robustft_20260724/upload_model.py" \
      --model "$ROOT/train/$arm" --arm "$arm" \
      --audit "$ROOT/hf_uploads/${arm}.json" > "$ROOT/logs/upload_${arm}.log" 2>&1
    find "$ROOT/train/$arm" -type f \
      \( -name optimizer.pt -o -name scheduler.pt -o -name rng_state.pth \) -delete
    find "$ROOT/train/$arm" -maxdepth 1 -type d -name 'checkpoint-*' \
      -prune -exec rm -rf {} +
  fi
done
date -Is > "$ROOT/EVAL_GENERATIONS_COMPLETE"
