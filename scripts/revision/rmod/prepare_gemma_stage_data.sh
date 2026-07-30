#!/usr/bin/env bash
set -euo pipefail

SOURCE=${SOURCE:?set SOURCE policy}
WORK=${WORK:?set WORK}
GPUS=${GPUS:-0,1,2,3}
NPROC=${NPROC:-4}
SCORE_SHARDS=${SCORE_SHARDS:-4}
SJ=/NHNHOME/AIPR/sjkim
P=$SJ/MNPO_rev_20260720
PY=$SJ/venv_clean/bin/python
BASE=google/gemma-2-2b-it
CACHE=$SJ/baseline_repair_1p5b_20260714/cache/huggingface/hub
IDX=6,7,8,9,10
NAMES_CSV=instruction_following,truthfulness,honesty,helpfulness,safety
mkdir -p "$WORK/pool/logs" "$WORK/scored" "$WORK/pairs"

export PYTHONPATH=$P
export HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface
export HF_HUB_CACHE=$CACHE HUGGINGFACE_HUB_CACHE=$CACHE
export TOKENIZERS_PARALLELISM=false MNPO_DISABLE_APEX=1 NCCL_SOCKET_IFNAME=lo

IFS=, read -r -a gpu <<< "$GPUS"
(( ${#gpu[@]} >= 4 )) || { echo "four GPUs required" >&2; exit 2; }

decode_one() {
  local g=$1 seeds=$2 split=$3
  CUDA_VISIBLE_DEVICES=$g "$PY" -u -m on_policy_data_gen.decode \
    --data_dir "$P/data/gemma2_ufb_part2_${split}.jsonl" --model "$SOURCE" --seeds $seeds \
    --output_dir "$WORK/pool/$split/gpu$g" --num_gpu 1 --temperature 0.8 --top_p 0.95 \
    --max_tokens 1024 --batch_size 512 --dtype bfloat16 --cache_dir "$CACHE" \
    > "$WORK/pool/logs/decode_${split}_gpu${g}.log" 2>&1
}

valid_output() {
  local path=$1 expected=$2
  [[ -s "$path" ]] && "$PY" - "$path" "$expected" <<'PY'
import json, sys
raise SystemExit(0 if len(json.load(open(sys.argv[1]))) == int(sys.argv[2]) else 1)
PY
}

decode_missing() {
  local g=$1 split=$2 expected=$3
  shift 3
  local missing=() seed
  for seed in "$@"; do
    valid_output "$WORK/pool/$split/gpu$g/output_${seed}.json" "$expected" || missing+=("$seed")
  done
  (( ${#missing[@]} == 0 )) || decode_one "$g" "${missing[*]}" "$split"
}

for split in train test; do
  if [[ ! -s "$WORK/pool/$split/merged/all_outputs.json" ]]; then
    [[ $split == train ]] && expected=19856 || expected=647
    decode_missing "${gpu[0]}" "$split" "$expected" 13 21 & p0=$!
    decode_missing "${gpu[1]}" "$split" "$expected" 42 & p1=$!
    decode_missing "${gpu[2]}" "$split" "$expected" 79 & p2=$!
    decode_missing "${gpu[3]}" "$split" "$expected" 100 & p3=$!
    wait "$p0" "$p1" "$p2" "$p3"
    mkdir -p "$WORK/pool/$split/merged"
    cp -f "$WORK/pool/$split"/gpu*/output_*.json "$WORK/pool/$split/merged/"
    "$PY" -m on_policy_data_gen.post_process --generation_file_dir "$WORK/pool/$split/merged"
  fi

  if [[ ! -s "$WORK/scored/${split}_safety.jsonl" ]]; then
    pids=()
    for shard in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=${gpu[$shard]} "$PY" -m on_policy_data_gen.rm_armo_multihead \
        --input_file "$WORK/pool/$split/merged/all_outputs.json" --output_dir "$WORK/scored" \
        --split "$split" --indices "$IDX" --names "$NAMES_CSV" --cache_dir "$CACHE" \
        --batch_size 8 --sample_batch_size 32 --num_shards "$SCORE_SHARDS" --shard_index "$shard" \
        > "$WORK/pool/logs/score_${split}_shard${shard}.log" 2>&1 & pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    for name in ${NAMES_CSV//,/ }; do
      while (( $(find "$WORK/scored" -maxdepth 1 -name "${split}_${name}.jsonl.shard*" | wc -l) < SCORE_SHARDS )); do sleep 10; done
      cat "$WORK/scored/${split}_${name}.jsonl.shard"* > "$WORK/scored/${split}_${name}.jsonl"
    done
  fi
done

if [[ ! -s "$WORK/pairs/train_ronpo.jsonl" ]]; then
  train_scores=(); test_scores=()
  for name in ${NAMES_CSV//,/ }; do
    train_scores+=("$name=$WORK/scored/train_${name}.jsonl")
    test_scores+=("$name=$WORK/scored/test_${name}.jsonl")
  done
  "$PY" -m mnpo_scripts.build_multi_objective_dataset --scored_files "${train_scores[@]}" \
    --mnpo_output "$WORK/pairs/train_mnpo_unused.jsonl" --ronpo_output "$WORK/pairs/train_ronpo.jsonl" \
    --merged_output "$WORK/pairs/train_merged.jsonl" --normalization minmax --ronpo_pair_strategy sigma \
    --ronpo_policy_pair_mode expected_relative_policy_vs_policy --ronpo_policy_samples_per_atom 1 \
    --adversary_steps 25 --adversary_alpha 1.0 --adversary_kappa 0.05 \
    --preference_scale 8.0 --pairs_per_prompt 2
  "$PY" -m mnpo_scripts.build_multi_objective_dataset --scored_files "${test_scores[@]}" \
    --mnpo_output "$WORK/pairs/test_mnpo_unused.jsonl" --ronpo_output "$WORK/pairs/test_ronpo.jsonl" \
    --merged_output "$WORK/pairs/test_merged.jsonl" --normalization minmax --ronpo_pair_strategy sigma \
    --ronpo_policy_pair_mode expected_relative_policy_vs_policy --ronpo_policy_samples_per_atom 1 \
    --adversary_steps 25 --adversary_alpha 1.0 --adversary_kappa 0.05 \
    --preference_scale 8.0 --pairs_per_prompt 2
fi

if [[ ! -f "$WORK/precomputed/dataset_dict.json" ]]; then
  CUDA_VISIBLE_DEVICES=$GPUS "$PY" -m accelerate.commands.launch --num_processes="$NPROC" \
    --main_process_port=30510 -m mnpo_scripts.precompute \
    --model_name_or_path "$SOURCE" --ref_model "$BASE" --history_paths "$SOURCE" \
    --train_dir "$WORK/pairs/train_ronpo.jsonl" --test_dir "$WORK/pairs/test_ronpo.jsonl" \
    --output_dir "$WORK/precomputed" --per_device_train_batch_size 8 --max_length 2048 \
    --max_prompt_length 1800 --apply_chat_template true --auto_insert_empty_system_msg false \
    --ronpo_target_mode none --sanity_check False > "$WORK/pool/logs/precompute.log" 2>&1
fi

test -f "$WORK/precomputed/dataset_dict.json"
date -Is > "$WORK/precomputed/READY"
