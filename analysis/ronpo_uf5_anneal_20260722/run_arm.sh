#!/usr/bin/env bash
set -euo pipefail

ARM=${1:?moving_anchor or stronger_signal}
GPUS=${GPUS:-0,1,2,3}
[[ "$ARM" == moving_anchor || "$ARM" == stronger_signal ]] || exit 2

SJ=/NHNHOME/AIPR/sjkim
P=$SJ/MNPO_rev_20260720
ROOT=$SJ/ronpo_uf5_anneal_20260722
PY=$SJ/venv_clean/bin/python
CACHE=$SJ/baseline_repair_1p5b_20260714/cache/huggingface/hub
BASE=google/gemma-2-2b-it
BASE_GEN=$SJ/rmod_20260720/radar/gens/base128/output_42.json
NAMES=(instruction_following truthfulness honesty helpfulness safety)
NAMES_CSV=instruction_following,truthfulness,honesty,helpfulness,safety
IDX=6,7,8,9,10
IFS=, read -r -a GPU <<< "$GPUS"
(( ${#GPU[@]} == 4 )) || { echo "four GPUs required" >&2; exit 2; }

export PYTHONPATH=$P
export HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface
export HF_HUB_CACHE=$CACHE HUGGINGFACE_HUB_CACHE=$CACHE
export TOKENIZERS_PARALLELISM=false MNPO_DISABLE_APEX=1 NCCL_SOCKET_IFNAME=lo
export WANDB_ENTITY=promotion-kim WANDB_PROJECT=mnpo WANDB_MODE=online WANDB_DIR=$ROOT/wandb
mkdir -p "$ROOT/$ARM" "$ROOT/wandb" "$ROOT/audit" "$ROOT/hf_uploads"

stamp() { date -Is; }
logfix() { printf '%s: %s\n' "$(stamp)" "$*" >> "$ROOT/fix_log.md"; }

valid_json_count() {
  "$PY" - "$1" "$2" <<'PY'
import json,sys
raise SystemExit(0 if len(json.load(open(sys.argv[1]))) == int(sys.argv[2]) else 1)
PY
}

decode() {
  local model=$1 temp=$2 seeds=$3 out=$4 split=$5 gpu=$6
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u -m on_policy_data_gen.decode \
    --data_dir "$P/data/gemma2_ufb_part2_${split}.jsonl" --model "$model" --seeds $seeds \
    --output_dir "$out" --num_gpu 1 --temperature "$temp" --top_p 0.95 \
    --max_tokens 1024 --batch_size 512 --dtype bfloat16 --cache_dir "$CACHE"
}

build_base_pool() {
  local shared=$ROOT/stronger_signal/shared_base
  [[ -s "$shared/READY" ]] && return
  for split in train test; do
    local expected=19856; [[ $split == test ]] && expected=647
    mkdir -p "$shared/$split"
    local pids=()
    for i in 0 1 2 3; do
      local seeds=(13 21 42 79)
      valid_json_count "$shared/$split/output_${seeds[$i]}.json" "$expected" ||
        decode "$BASE" 0.8 "${seeds[$i]}" "$shared/$split" "$split" "${GPU[$i]}" \
          > "$shared/${split}_${seeds[$i]}.log" 2>&1 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
  done
  stamp > "$shared/READY"
}

decode_pool() {
  local stage=$1 parent=$2 work=$3
  [[ -s "$work/pool/READY" ]] && return
  if [[ $ARM == stronger_signal ]]; then build_base_pool; fi
  for split in train test; do
    local expected=19856; [[ $split == test ]] && expected=647
    local merged=$work/pool/$split/merged
    mkdir -p "$merged" "$work/logs"
    if [[ $ARM == moving_anchor ]]; then
      local seeds=(13 21 42 79 100) groups=('13 21' 42 79 100)
      local pids=()
      for i in 0 1 2 3; do
        local out=$work/pool/$split/parent_gpu${GPU[$i]}
        local need=0 s
        for s in ${groups[$i]}; do valid_json_count "$out/output_${s}.json" "$expected" || need=1; done
        (( need == 0 )) || decode "$parent" 0.8 "${groups[$i]}" "$out" "$split" "${GPU[$i]}" \
          > "$work/logs/decode_${split}_gpu${GPU[$i]}.log" 2>&1 &
        pids+=("$!")
      done
      for pid in "${pids[@]}"; do wait "$pid"; done
      cp -f "$work/pool/$split"/parent_gpu*/output_*.json "$merged/"
    else
      cp -f "$ROOT/stronger_signal/shared_base/$split"/output_*.json "$merged/"
      local pids=()
      for i in 0 1 2 3; do
        local seed=$((100+i)) out=$work/pool/$split/parent_standard
        valid_json_count "$out/output_${seed}.json" "$expected" ||
          decode "$parent" 0.8 "$seed" "$out" "$split" "${GPU[$i]}" \
            > "$work/logs/decode_${split}_parent_${seed}.log" 2>&1 &
        pids+=("$!")
      done
      for pid in "${pids[@]}"; do wait "$pid"; done
      pids=()
      for i in 0 1; do
        local seed=$((104+i)) out=$work/pool/$split/parent_hot
        valid_json_count "$out/output_${seed}.json" "$expected" ||
          decode "$parent" 1.0 "$seed" "$out" "$split" "${GPU[$i]}" \
            > "$work/logs/decode_${split}_parent_hot_${seed}.log" 2>&1 &
        pids+=("$!")
      done
      for pid in "${pids[@]}"; do wait "$pid"; done
      cp -f "$work/pool/$split/parent_standard"/output_*.json "$merged/"
      cp -f "$work/pool/$split/parent_hot"/output_*.json "$merged/"
    fi
    "$PY" -m on_policy_data_gen.post_process --generation_file_dir "$merged" \
      > "$work/logs/merge_${split}.log" 2>&1
    valid_json_count "$merged/all_outputs.json" "$expected" || {
      count=$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$merged/all_outputs.json")
      logfix "$ARM stage $stage $split retained $count/$expected prompts after identical-response filtering"
    }
  done
  stamp > "$work/pool/READY"
}

score_pool() {
  local stage=$1 work=$2
  [[ -s "$work/scored/READY" ]] && return
  mkdir -p "$work/scored" "$work/logs"
  for split in train test; do
    local pids=()
    for shard in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=${GPU[$shard]} "$PY" -m on_policy_data_gen.rm_armo_multihead \
        --input_file "$work/pool/$split/merged/all_outputs.json" --output_dir "$work/scored" \
        --split "$split" --indices "$IDX" --names "$NAMES_CSV" --cache_dir "$CACHE" \
        --batch_size 8 --sample_batch_size 32 --num_shards 4 --shard_index "$shard" \
        > "$work/logs/score_${split}_shard${shard}.log" 2>&1 & pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    for name in "${NAMES[@]}"; do
      : > "$work/scored/${split}_${name}.jsonl"
      for shard in 0 1 2 3; do
        test -s "$work/scored/${split}_${name}.jsonl.shard${shard}"
        sed -n '1,$p' "$work/scored/${split}_${name}.jsonl.shard${shard}" \
          >> "$work/scored/${split}_${name}.jsonl"
      done
    done
  done
  stamp > "$work/scored/READY"
}

prepare_stage() {
  local stage=$1 parent=$2 ref=$3 work=$4
  [[ -s "$work/precomputed_os/READY" ]] && return
  decode_pool "$stage" "$parent" "$work"
  score_pool "$stage" "$work"
  mkdir -p "$work/pairs" "$work/logs"
  local train_scores=() test_scores=() name
  for name in "${NAMES[@]}"; do
    train_scores+=("$name=$work/scored/train_${name}.jsonl")
    test_scores+=("$name=$work/scored/test_${name}.jsonl")
  done
  if [[ ! -s "$work/pairs/train_ronpo.jsonl" ]]; then
    "$PY" -m mnpo_scripts.build_multi_objective_dataset --scored_files "${train_scores[@]}" \
      --mnpo_output "$work/pairs/train_unused.jsonl" --ronpo_output "$work/pairs/train_ronpo.jsonl" \
      --normalization minmax --ronpo_pair_strategy sigma \
      --ronpo_policy_pair_mode expected_relative_policy_vs_policy --ronpo_policy_samples_per_atom 1 \
      --adversary_steps 25 --adversary_alpha 1.0 --adversary_kappa 0.05 \
      --preference_scale 8.0 --pairs_per_prompt 2 > "$work/logs/build_train_pairs.log" 2>&1
    "$PY" -m mnpo_scripts.build_multi_objective_dataset --scored_files "${test_scores[@]}" \
      --mnpo_output "$work/pairs/test_unused.jsonl" --ronpo_output "$work/pairs/test_ronpo.jsonl" \
      --normalization minmax --ronpo_pair_strategy sigma \
      --ronpo_policy_pair_mode expected_relative_policy_vs_policy --ronpo_policy_samples_per_atom 1 \
      --adversary_steps 25 --adversary_alpha 1.0 --adversary_kappa 0.05 \
      --preference_scale 8.0 --pairs_per_prompt 2 > "$work/logs/build_test_pairs.log" 2>&1
  fi
  if [[ ! -f "$work/precomputed_raw/dataset_dict.json" ]]; then
    CUDA_VISIBLE_DEVICES=$GPUS "$PY" -m accelerate.commands.launch --num_processes=4 \
      --main_process_port=$((31100+stage)) -m mnpo_scripts.precompute \
      --model_name_or_path "$parent" --ref_model "$ref" --history_paths "$parent" \
      --train_dir "$work/pairs/train_ronpo.jsonl" --test_dir "$work/pairs/test_ronpo.jsonl" \
      --output_dir "$work/precomputed_raw" --per_device_train_batch_size 8 \
      --max_length 2048 --max_prompt_length 1800 --apply_chat_template true \
      --auto_insert_empty_system_msg false --ronpo_target_mode none --sanity_check False \
      > "$work/logs/precompute.log" 2>&1
  fi
  if [[ ! -f "$work/precomputed_os/dataset_dict.json" ]]; then
    "$PY" -m mnpo_scripts.build_os_ronpo_targets --input_dir "$work/precomputed_raw" \
      --output_dir "$work/precomputed_os" --kappas 0.05 --num_proc 12 \
      > "$work/logs/build_os_targets.log" 2>&1
  fi
  stamp > "$work/precomputed_os/READY"
}

train_stage() {
  local stage=$1 parent=$2 work=$3
  local out=$work/train lr=5.0e-7
  [[ $ARM == stronger_signal ]] && lr=1.0e-6
  [[ -s "$out/all_results.json" ]] && return
  mkdir -p "$out" "$work/logs"
  local resume=() last
  last=$(find "$out" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null | sort -V | tail -1)
  [[ -n $last ]] && resume=(--resume_from_checkpoint="$out/$last")
  CUDA_VISIBLE_DEVICES=$GPUS "$PY" -m accelerate.commands.launch \
    --config_file "$P/accelerate_configs/multi_gpu.yaml" --num_processes=4 \
    --main_process_port=$((31200+stage)) -m mnpo_scripts.run_mnpo \
    "$P/training_configs/ronpo/qwen2.5-1.5b-instruct-ronpo-multiobj-iter1.yaml" \
    --model_name_or_path="$parent" --dataset_mixer="$work/precomputed_os:1.0" \
    --loss_type=ronpo --ronpo_alpha=1.0 --ronpo_tau=0.05 --ronpo_target_column=target_os_k0p05 \
    --max_history_t=1 --history_weights=1.0 --learning_rate="$lr" --warmup_ratio=0.1 \
    --num_train_epochs=1 --seed=42 --max_steps=1800 --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=4 --gradient_checkpointing=true --max_length=2048 \
    --max_prompt_length=1800 --do_eval=false --eval_strategy=no --generate_during_eval=false \
    --save_strategy=steps --save_steps=600 --save_total_limit=1 --logging_steps=10 \
    --output_dir="$out" --run_name="ronpo-uf5-${ARM}-stage${stage}-s42" "${resume[@]}" \
    > "$work/logs/train.log" 2>&1
  test -s "$out/all_results.json"
}

gate_stage() {
  local stage=$1 work=$2 model=$work/train
  mkdir -p "$work/eval" "$work/logs"
  if [[ ! -s "$work/eval/output_42.json" ]]; then
    CUDA_VISIBLE_DEVICES=${GPU[0]} "$PY" -u -m on_policy_data_gen.decode \
      --data_dir "$P/data/gemma2_ufb_part2_test.jsonl" --model "$model" --seeds 42 \
      --output_dir "$work/eval" --num_gpu 1 --temperature 0.8 --top_p 0.95 \
      --max_tokens 128 --batch_size 256 --dtype bfloat16 --cache_dir "$CACHE" \
      > "$work/logs/decode_eval.log" 2>&1
  fi
  set +e
  "$PY" "$P/scripts/revision/flagship/stability_gate_corrected.py" \
    --base "$BASE_GEN" --candidate "$work/eval/output_42.json" --expected-records 647 \
    --min-length-ratio 0.33 --max-length-ratio 2.0 --max-repeat-run 20 \
    --output "$work/eval/stability_gate.json" > "$work/logs/stability_gate.log" 2>&1
  local rc=$?
  set -e
  if (( rc != 0 )); then
    logfix "$ARM stage $stage failed the locked reward-blind stability gate (rc=$rc); chain stopped"
    stamp > "$work/eval/GATE_FAILED"
    return "$rc"
  fi
  stamp > "$work/eval/GATE_PASSED"
}

upload_stage() {
  local stage=$1 work=$2
  "$PY" "$P/analysis/ronpo_uf5_anneal_20260722/upload_stage.py" \
    --model "$work/train" --arm "$ARM" --stage "$stage" \
    --audit "$ROOT/hf_uploads/${ARM}_stage${stage}.json" \
    > "$work/logs/hf_upload.log" 2>&1
}

upload_pids=()
for stage in 1 2 3 4; do
  work=$ROOT/$ARM/stage$stage
  parent=$BASE
  (( stage > 1 )) && parent=$ROOT/$ARM/stage$((stage-1))/train
  ref=$BASE
  [[ $ARM == moving_anchor && $stage -gt 1 ]] && ref=$parent

  if [[ $stage == 1 && $ARM == moving_anchor ]]; then
    # The locked Base pool/logps are read-only and byte-reused; training is new and uses 1,800 steps.
    mkdir -p "$work" "$work/logs"
    if [[ ! -e "$work/precomputed_os" ]]; then ln -s "$SJ/ronpo_gemma_20260720/precomputed_kappa" "$work/precomputed_os"; fi
  else
    prepare_stage "$stage" "$parent" "$ref" "$work"
  fi
  train_stage "$stage" "$parent" "$work"
  gate_stage "$stage" "$work"
  upload_stage "$stage" "$work" & upload_pids+=("$!")
  stamp > "$work/STAGE_COMPLETE"
done

for pid in "${upload_pids[@]}"; do wait "$pid"; done
for stage in 1 2 3 4; do
  work=$ROOT/$ARM/stage$stage
  test -s "$ROOT/hf_uploads/${ARM}_stage${stage}.json"
  find "$work/train" -type f \( -name optimizer.pt -o -name scheduler.pt -o -name rng_state.pth \) -delete
  find "$work/train" -maxdepth 1 -type d -name 'checkpoint-*' -prune -exec rm -rf {} +
done
stamp > "$ROOT/$ARM/ARM_COMPLETE"

