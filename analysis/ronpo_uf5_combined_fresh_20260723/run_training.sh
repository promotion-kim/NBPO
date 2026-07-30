#!/usr/bin/env bash
set -euo pipefail

SJ=/NHNHOME/AIPR/sjkim
P=$SJ/MNPO_rev_20260720
ROOT=$SJ/ronpo_uf5_combined_fresh_20260723
PY=$SJ/venv_clean/bin/python
CACHE=$SJ/baseline_repair_1p5b_20260714/cache/huggingface/hub
BASE=google/gemma-2-2b-it
BASE_GEN=$SJ/rmod_20260720/radar/gens/base128/output_42.json
BASE_POOL=$ROOT/shared/base
PARENT_ROOT=$ROOT/parents/ss_s4
PARENT=$PARENT_ROOT/stronger_signal/stage4
NAMES=(instruction_following truthfulness honesty helpfulness safety)
NAMES_CSV=instruction_following,truthfulness,honesty,helpfulness,safety
IDX=6,7,8,9,10

export PYTHONPATH=$P
export HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface
export HF_HUB_CACHE=$CACHE HUGGINGFACE_HUB_CACHE=$CACHE
export TOKENIZERS_PARALLELISM=false MNPO_DISABLE_APEX=1 NCCL_SOCKET_IFNAME=lo
export WANDB_ENTITY=promotion-kim WANDB_PROJECT=mnpo WANDB_MODE=online WANDB_DIR=$ROOT/wandb
mkdir -p "$ROOT"/{wandb,audit,hf_uploads,logs,shared}

stamp() { date -Is; }
logfix() { printf '%s: %s\n' "$(stamp)" "$*" >> "$ROOT/fix_log.md"; }
valid_count() { "$PY" - "$1" "$2" <<'PY'
import json,sys
raise SystemExit(0 if len(json.load(open(sys.argv[1]))) == int(sys.argv[2]) else 1)
PY
}

if [[ ! -s "$PARENT/model.safetensors" ]]; then
  "$PY" "$P/analysis/ronpo_uf5_combined_fresh_20260723/download_parent.py" --output "$PARENT_ROOT" \
    > "$ROOT/logs/download_parent.log" 2>&1
fi

build_base_pool() {
  for split in train test; do
    expected=19856; [[ $split == test ]] && expected=647
    mkdir -p "$BASE_POOL/$split"
    pids=()
    seeds=(13 21 42 79)
    for i in 0 1 2 3; do
      seed=${seeds[$i]}
      if ! valid_count "$BASE_POOL/$split/output_${seed}.json" "$expected"; then
        decode "$BASE" 0.8 "$seed" "$BASE_POOL/$split" "$split" "$i" \
          > "$ROOT/logs/decode_base_${split}_${seed}.log" 2>&1 & pids+=("$!")
      fi
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    for seed in "${seeds[@]}"; do valid_count "$BASE_POOL/$split/output_${seed}.json" "$expected"; done
  done
}

decode() {
  local model=$1 temp=$2 seed=$3 out=$4 split=$5 gpu=$6
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u -m on_policy_data_gen.decode \
    --data_dir "$P/data/gemma2_ufb_part2_${split}.jsonl" --model "$model" --seeds "$seed" \
    --output_dir "$out" --num_gpu 1 --temperature "$temp" --top_p 0.95 \
    --max_tokens 1024 --batch_size 512 --dtype bfloat16 --cache_dir "$CACHE"
}

if [[ ! -s "$BASE_POOL/train/output_13.json" ]]; then
  logfix "Previously uploaded/pruned stronger-signal Base response pool was absent; regenerated the four preregistered Base seeds with unchanged decode settings."
  build_base_pool
fi
for split in train test; do
  expected=19856; [[ $split == test ]] && expected=647
  for seed in 13 21 42 79; do
    valid_count "$BASE_POOL/$split/output_${seed}.json" "$expected"
  done
done

build_pool() {
  local parent=$1 work=$2
  [[ -s "$work/pool/READY" ]] && return
  mkdir -p "$work/pool" "$work/logs"
  for split in train test; do
    expected=19856; [[ $split == test ]] && expected=647
    mkdir -p "$work/pool/$split/merged" "$work/pool/$split/parent_standard" "$work/pool/$split/parent_hot"
    pids=()
    for i in 0 1 2 3; do
      seed=$((100+i))
      if ! valid_count "$work/pool/$split/parent_standard/output_${seed}.json" "$expected"; then
        decode "$parent" 0.8 "$seed" "$work/pool/$split/parent_standard" "$split" "$i" \
          > "$work/logs/decode_${split}_parent_${seed}.log" 2>&1 & pids+=("$!")
      fi
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    pids=()
    for i in 0 1; do
      seed=$((104+i))
      if ! valid_count "$work/pool/$split/parent_hot/output_${seed}.json" "$expected"; then
        decode "$parent" 1.0 "$seed" "$work/pool/$split/parent_hot" "$split" "$i" \
          > "$work/logs/decode_${split}_parent_hot_${seed}.log" 2>&1 & pids+=("$!")
      fi
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    rm -f "$work/pool/$split/merged"/output_*.json "$work/pool/$split/merged/all_outputs.json"
    cp "$BASE_POOL/$split"/output_*.json "$work/pool/$split/merged/"
    cp "$work/pool/$split/parent_standard"/output_*.json "$work/pool/$split/merged/"
    cp "$work/pool/$split/parent_hot"/output_*.json "$work/pool/$split/merged/"
    "$PY" -m on_policy_data_gen.post_process --generation_file_dir "$work/pool/$split/merged" \
      > "$work/logs/merge_${split}.log" 2>&1
  done
  stamp > "$work/pool/READY"
}

score_pool() {
  local work=$1
  [[ -s "$work/scored/READY" ]] && return
  mkdir -p "$work/scored" "$work/logs"
  for split in train test; do
    pids=()
    for shard in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$shard "$PY" -m on_policy_data_gen.rm_armo_multihead \
        --input_file "$work/pool/$split/merged/all_outputs.json" --output_dir "$work/scored" \
        --split "$split" --indices "$IDX" --names "$NAMES_CSV" --cache_dir "$CACHE" \
        --batch_size 8 --sample_batch_size 32 --num_shards 4 --shard_index "$shard" \
        > "$work/logs/score_${split}_shard${shard}.log" 2>&1 & pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    for name in "${NAMES[@]}"; do
      : > "$work/scored/${split}_${name}.jsonl"
      for shard in 0 1 2 3; do cat "$work/scored/${split}_${name}.jsonl.shard${shard}" >> "$work/scored/${split}_${name}.jsonl"; done
    done
  done
  stamp > "$work/scored/READY"
}

build_pairs() {
  local work=$1
  [[ -s "$work/pairs/READY" ]] && return
  mkdir -p "$work/pairs" "$work/logs"
  train_scores=(); test_scores=()
  for name in "${NAMES[@]}"; do
    train_scores+=("$name=$work/scored/train_${name}.jsonl")
    test_scores+=("$name=$work/scored/test_${name}.jsonl")
  done
  "$PY" -m mnpo_scripts.build_multi_objective_dataset --scored_files "${train_scores[@]}" \
    --mnpo_output "$work/pairs/train_unused.jsonl" --ronpo_output "$work/pairs/train_ronpo.jsonl" \
    --normalization minmax --ronpo_pair_strategy sigma --ronpo_policy_pair_mode expected_relative_policy_vs_policy \
    --ronpo_policy_samples_per_atom 1 --adversary_steps 25 --adversary_alpha 1.0 --adversary_kappa 0.05 \
    --preference_scale 8.0 --pairs_per_prompt 2 > "$work/logs/build_train_pairs.log" 2>&1
  "$PY" -m mnpo_scripts.build_multi_objective_dataset --scored_files "${test_scores[@]}" \
    --mnpo_output "$work/pairs/test_unused.jsonl" --ronpo_output "$work/pairs/test_ronpo.jsonl" \
    --normalization minmax --ronpo_pair_strategy sigma --ronpo_policy_pair_mode expected_relative_policy_vs_policy \
    --ronpo_policy_samples_per_atom 1 --adversary_steps 25 --adversary_alpha 1.0 --adversary_kappa 0.05 \
    --preference_scale 8.0 --pairs_per_prompt 2 > "$work/logs/build_test_pairs.log" 2>&1
  stamp > "$work/pairs/READY"
}

precompute() {
  local parent=$1 ref=$2 source=$3 out=$4 gpus=$5 port=$6 processes=$7
  mkdir -p "$out/logs"
  if [[ ! -f "$out/precomputed_raw/dataset_dict.json" ]]; then
    CUDA_VISIBLE_DEVICES=$gpus "$PY" -m accelerate.commands.launch --num_processes="$processes" \
      --main_process_port="$port" -m mnpo_scripts.precompute --model_name_or_path "$parent" \
      --ref_model "$ref" --history_paths "$parent" --train_dir "$source/pairs/train_ronpo.jsonl" \
      --test_dir "$source/pairs/test_ronpo.jsonl" --output_dir "$out/precomputed_raw" \
      --per_device_train_batch_size 8 --max_length 2048 --max_prompt_length 1800 \
      --apply_chat_template true --auto_insert_empty_system_msg false --ronpo_target_mode none \
      --sanity_check False > "$out/logs/precompute.log" 2>&1
  fi
  if [[ ! -f "$out/precomputed_os/dataset_dict.json" ]]; then
    "$PY" -m mnpo_scripts.build_os_ronpo_targets --input_dir "$out/precomputed_raw" \
      --output_dir "$out/precomputed_os" --kappas 0.05 --num_proc 8 > "$out/logs/build_targets.log" 2>&1
  fi
  stamp > "$out/precomputed_os/READY"
}

train() {
  local arm=$1 stage=$2 parent=$3 work=$4 gpus=$5 port=$6 processes=$7 accum=$8
  mkdir -p "$work/train" "$work/logs"
  resume=(); last=$(find "$work/train" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null | sort -V | tail -1)
  [[ -n $last ]] && resume=(--resume_from_checkpoint="$work/train/$last")
  CUDA_VISIBLE_DEVICES=$gpus "$PY" -m accelerate.commands.launch \
    --config_file "$P/accelerate_configs/multi_gpu.yaml" --num_processes="$processes" \
    --main_process_port="$port" -m mnpo_scripts.run_mnpo \
    "$P/training_configs/ronpo/qwen2.5-1.5b-instruct-ronpo-multiobj-iter1.yaml" \
    --model_name_or_path="$parent" --dataset_mixer="$work/precomputed_os:1.0" \
    --loss_type=ronpo --ronpo_alpha=1.0 --ronpo_tau=0.05 --ronpo_target_column=target_os_k0p05 \
    --max_history_t=1 --history_weights=1.0 --learning_rate=1.0e-6 --warmup_ratio=0.1 \
    --num_train_epochs=1 --seed=42 --max_steps=1800 --per_device_train_batch_size=1 \
    --gradient_accumulation_steps="$accum" --gradient_checkpointing=true --max_length=2048 \
    --max_prompt_length=1800 --do_eval=false --eval_strategy=no --generate_during_eval=false \
    --save_strategy=steps --save_steps=600 --save_total_limit=1 --logging_steps=10 \
    --output_dir="$work/train" --run_name="ronpo-uf5-${arm}-stage${stage}-s42" "${resume[@]}" \
    > "$work/logs/train.log" 2>&1
  test -s "$work/train/all_results.json"
}

gate_upload() {
  local arm=$1 stage=$2 work=$3 gpu=$4
  mkdir -p "$work/eval" "$work/logs"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u -m on_policy_data_gen.decode \
    --data_dir "$P/data/gemma2_ufb_part2_test.jsonl" --model "$work/train" --seeds 42 \
    --output_dir "$work/eval" --num_gpu 1 --temperature 0.8 --top_p 0.95 --max_tokens 128 \
    --batch_size 256 --dtype bfloat16 --cache_dir "$CACHE" > "$work/logs/decode_eval.log" 2>&1
  set +e
  "$PY" "$P/scripts/revision/flagship/stability_gate_corrected.py" --base "$BASE_GEN" \
    --candidate "$work/eval/output_42.json" --expected-records 647 --min-length-ratio 0.33 \
    --max-length-ratio 2.0 --max-repeat-run 20 --output "$work/eval/stability_gate.json" \
    > "$work/logs/stability_gate.log" 2>&1
  rc=$?; set -e
  if (( rc != 0 )); then
    logfix "$arm stage $stage failed locked stability gate rc=$rc"
    stamp > "$work/eval/GATE_FAILED"
    return "$rc"
  fi
  stamp > "$work/eval/GATE_PASSED"
  "$PY" "$P/analysis/ronpo_uf5_combined_fresh_20260723/upload_stage.py" \
    --model "$work/train" --arm "$arm" --stage "$stage" \
    --audit "$ROOT/hf_uploads/${arm}_stage${stage}.json" > "$work/logs/hf_upload.log" 2>&1
  find "$work/train" -type f \( -name optimizer.pt -o -name scheduler.pt -o -name rng_state.pth \) -delete
  find "$work/train" -maxdepth 1 -type d -name 'checkpoint-*' -prune -exec rm -rf {} +
}

if [[ ${MODE:-full} == post_s5 ]]; then
  # Resume only transport and Stage 6 after both locked Stage-5 final
  # checkpoints completed and passed the full stability gate.
  (
    for arm in fixed_base combined; do
      work=$ROOT/$arm/stage5
      test -s "$work/eval/GATE_PASSED"
      "$PY" "$P/analysis/ronpo_uf5_combined_fresh_20260723/upload_stage.py" \
        --model "$work/train" --arm "$arm" --stage 5 \
        --audit "$ROOT/hf_uploads/${arm}_stage5.json" > "$work/logs/hf_upload_retry.log" 2>&1
      find "$work/train" -type f \( -name optimizer.pt -o -name scheduler.pt -o -name rng_state.pth \) -delete
      find "$work/train" -maxdepth 1 -type d -name 'checkpoint-*' -prune -exec rm -rf {} +
    done
    stamp > "$ROOT/S5_UPLOADS_COMPLETE"
  ) & s5_upload_pid=$!

  S6=$ROOT/combined/stage6
  PARENT6=$ROOT/combined/stage5/train
  build_pool "$PARENT6" "$S6"
  score_pool "$S6"
  build_pairs "$S6"
  precompute "$PARENT6" "$PARENT6" "$S6" "$S6" 0,1,2,3 32161 4
  train combined 6 "$PARENT6" "$S6" 0,1,2,3 32261 4 4
  gate_upload combined 6 "$S6" 0
  wait "$s5_upload_pid"
  stamp > "$ROOT/TRAINING_COMPLETE"
  exit 0
fi

if [[ ${MODE:-full} == stage7 ]]; then
  while [[ ! -s "$ROOT/combined/stage6/eval/GATE_PASSED" && ! -s "$ROOT/combined/stage6/eval/GATE_FAILED" ]]; do sleep 30; done
  if [[ -s "$ROOT/combined/stage6/eval/GATE_FAILED" ]]; then
    logfix "Stage 7 was not run because Stage 6 failed the locked stability gate."
    stamp > "$ROOT/CHAIN_TERMINAL"
    exit 0
  fi
  S7_CUTOFF=$(date -d '2026-07-23 05:15:00 +0900' +%s)
  if (( $(date +%s) < S7_CUTOFF )); then
    S7=$ROOT/combined/stage7
    PARENT7=$ROOT/combined/stage6/train
    build_pool "$PARENT7" "$S7"
    score_pool "$S7"
    build_pairs "$S7"
    precompute "$PARENT7" "$PARENT7" "$S7" "$S7" 0,1,2,3 32171 4
    train combined 7 "$PARENT7" "$S7" 0,1,2,3 32271 4 4
    gate_upload combined 7 "$S7" 0
  else
    mkdir -p "$ROOT/combined/stage7"
    logfix "Stage 7 was not started because the preregistered 05:15 KST cutoff had passed."
    stamp > "$ROOT/combined/stage7/SKIPPED_TIME_CUTOFF"
  fi
  stamp > "$ROOT/CHAIN_TERMINAL"
  exit 0
fi

# Common strong S5 pool, one scoring pass, and one row set.
S5=$ROOT/shared/stage5
build_pool "$PARENT" "$S5"
score_pool "$S5"
build_pairs "$S5"

# Only the reference log-probability differs between the preregistered arms.
precompute "$PARENT" "$BASE" "$S5" "$ROOT/fixed_base/stage5" 0,1 32151 2 & p0=$!
precompute "$PARENT" "$PARENT" "$S5" "$ROOT/combined/stage5" 2,3 32152 2 & p1=$!
wait "$p0"; wait "$p1"

train fixed_base 5 "$PARENT" "$ROOT/fixed_base/stage5" 0,1 32251 2 8 & t0=$!
train combined 5 "$PARENT" "$ROOT/combined/stage5" 2,3 32252 2 8 & t1=$!
wait "$t0"; wait "$t1"
gate_upload fixed_base 5 "$ROOT/fixed_base/stage5" 0 & g0=$!
gate_upload combined 5 "$ROOT/combined/stage5" 1 & g1=$!
wait "$g0"; wait "$g1"

# Primary S6 chain: fresh heterogeneous pool with the S5 policy as parent and prox center.
S6=$ROOT/combined/stage6
PARENT6=$ROOT/combined/stage5/train
build_pool "$PARENT6" "$S6"
score_pool "$S6"
build_pairs "$S6"
precompute "$PARENT6" "$PARENT6" "$S6" "$S6" 0,1,2,3 32161 4
train combined 6 "$PARENT6" "$S6" 0,1,2,3 32261 4 4
gate_upload combined 6 "$S6" 0
stamp > "$ROOT/TRAINING_COMPLETE"
