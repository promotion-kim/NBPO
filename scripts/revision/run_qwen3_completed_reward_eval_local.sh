#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/sjkim/MNPO}"
# shellcheck source=../setup_ext_cache.sh
source "$PROJECT_ROOT/scripts/setup_ext_cache.sh"

PYTHON_TRAIN="${PYTHON_TRAIN:-/home/sjkim/anaconda3/envs/ronpo-rev/bin/python}"
PYTHON_INFER="${PYTHON_INFER:-/home/sjkim/anaconda3/envs/mnpo_infer/bin/python}"
PYTHON_RM="${PYTHON_RM:-/home/sjkim/anaconda3/envs/mnpo_infer/bin/python}"
PYTHON_ARMO="${PYTHON_ARMO:-/home/sjkim/anaconda3/envs/mnpo_train/bin/python}"

EXP_ROOT="${EXP_ROOT:-/ext_hdd/sjkim/mnpo/revision_qwen3_8b/full_iter1}"
EVAL_FILE="${EVAL_FILE:-$PROJECT_ROOT/data/gemma2_ufb_part2_test.jsonl}"
WORK_DIR="${WORK_DIR:-$EXP_ROOT/eval/local_rm_completed_dpo_odin2_$(date +%Y%m%d_%H%M%S)}"
GEN_DIR="$WORK_DIR/generations"
SCORED_DIR="$WORK_DIR/scored"
RESULT_DIR="$WORK_DIR/results"
LOG_DIR="$WORK_DIR/logs"
SEED="${SEED:-42}"
GPU="${GPU:-1}"
MODEL_ID="${MODEL_ID:-/ext_hdd/sjkim/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"

mkdir -p "$GEN_DIR" "$SCORED_DIR" "$RESULT_DIR" "$LOG_DIR"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

MODEL_NAMES=(
  base
  dpo_b0p01
  dpo_b0p05
)
MODEL_PATHS=(
  "$MODEL_ID"
  "$EXP_ROOT/train/dpo_avg_beta0p01_s42_odin2"
  "$EXP_ROOT/train/dpo_avg_beta0p05_s42_odin2"
)

for model_path in "${MODEL_PATHS[@]}"; do
  test -f "$model_path/config.json"
  test -f "$model_path/model.safetensors.index.json"
done

echo "[start] local RM eval for completed odin2 DPO checkpoints $(date -Is)"
echo "[work] $WORK_DIR"
echo "[eval] $EVAL_FILE"
nvidia-smi || true

run_decode() {
  local name="$1"
  local model="$2"
  local out_dir="$GEN_DIR/$name"
  local out_file="$out_dir/output_${SEED}.json"
  mkdir -p "$out_dir"
  if [[ -s "$out_file" && "${FORCE_DECODE:-0}" != "1" ]]; then
    echo "[decode:$name] reuse $out_file"
    return
  fi
  echo "[decode:$name] gpu=$GPU model=$model $(date -Is)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_TRAIN" -u "$PROJECT_ROOT/scripts/revision/decode_transformers_non_thinking.py" \
    --data_dir "$EVAL_FILE" \
    --model "$model" \
    --seed "$SEED" \
    --output_dir "$out_dir" \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_new_tokens 4096 \
    --batch_size "${DECODE_BATCH_SIZE:-2}" \
    --attn_implementation "${DECODE_ATTN_IMPLEMENTATION:-sdpa}" \
    --cache_dir "$CACHE_DIR" \
    >"$LOG_DIR/decode_${name}.log" 2>&1
  "$PYTHON_TRAIN" "$PROJECT_ROOT/scripts/revision/check_qwen3_outputs.py" \
    "$out_file" \
    --max-mean-words "${GEN_MAX_MEAN_WORDS:-900}" \
    --max-any-words "${GEN_MAX_ANY_WORDS:-4096}" \
    > "$RESULT_DIR/${name}_length_check.json"
}

merge_args=()
for i in "${!MODEL_NAMES[@]}"; do
  run_decode "${MODEL_NAMES[$i]}" "${MODEL_PATHS[$i]}"
  merge_args+=("${MODEL_NAMES[$i]}=$GEN_DIR/${MODEL_NAMES[$i]}/output_${SEED}.json")
done

"$PYTHON_TRAIN" -m mnpo_scripts.merge_model_generations \
  --generations "${merge_args[@]}" \
  --output_file "$WORK_DIR/merged_model_generations.json" \
  >"$LOG_DIR/merge.log" 2>&1

expected_samples() {
  "$PYTHON_TRAIN" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' \
    "$WORK_DIR/merged_model_generations.json"
}

score_complete() {
  local output_file="$1"
  [[ -s "$output_file" ]] || return 1
  [[ "$(wc -l < "$output_file")" == "$(expected_samples)" ]]
}

if [[ "${FORCE_SCORE:-0}" == "1" ]] || ! score_complete "$SCORED_DIR/eval_skywork.jsonl"; then
  echo "[score:skywork] $(date -Is)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_RM" -u -m on_policy_data_gen.rm_skywork \
    --model_name Skywork/Skywork-Reward-V2-Llama-3.1-8B \
    --input_file "$WORK_DIR/merged_model_generations.json" \
    --output_file "$SCORED_DIR/eval_skywork.jsonl" \
    --cache_dir "$CACHE_DIR" \
    --batch_size "${SKYWORK_BATCH_SIZE:-4}" \
    --sample_batch_size "${SKYWORK_SAMPLE_BATCH_SIZE:-8}" \
    --attn_implementation "${SKYWORK_ATTN_IMPLEMENTATION:-sdpa}" \
    >"$LOG_DIR/score_skywork.log" 2>&1
fi

if [[ "${FORCE_SCORE:-0}" == "1" ]] || ! score_complete "$SCORED_DIR/eval_athene.jsonl"; then
  echo "[score:athene] $(date -Is)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_RM" -u -m on_policy_data_gen.rm_athene \
    --input_file "$WORK_DIR/merged_model_generations.json" \
    --output_file "$SCORED_DIR/eval_athene.jsonl" \
    --cache_dir "$CACHE_DIR" \
    --batch_size "${ATHENE_BATCH_SIZE:-4}" \
    --sample_batch_size "${ATHENE_SAMPLE_BATCH_SIZE:-8}" \
    >"$LOG_DIR/score_athene.log" 2>&1
fi

if [[ "${FORCE_SCORE:-0}" == "1" ]] || ! score_complete "$SCORED_DIR/eval_armo.jsonl"; then
  echo "[score:armo] $(date -Is)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_ARMO" -u -m on_policy_data_gen.rm_armo \
    --input_file "$WORK_DIR/merged_model_generations.json" \
    --output_file "$SCORED_DIR/eval_armo.jsonl" \
    --cache_dir "$CACHE_DIR" \
    --batch_size "${ARMO_BATCH_SIZE:-4}" \
    --sample_batch_size "${ARMO_SAMPLE_BATCH_SIZE:-8}" \
    --max_seq_length 4096 \
    >"$LOG_DIR/score_armo.log" 2>&1
fi

"$PYTHON_TRAIN" -m mnpo_scripts.evaluate_multi_objective_models \
  --scored_files \
    "skywork=$SCORED_DIR/eval_skywork.jsonl" \
    "athene=$SCORED_DIR/eval_athene.jsonl" \
    "armo=$SCORED_DIR/eval_armo.jsonl" \
  --output_dir "$RESULT_DIR" \
  --baseline_model base \
  >"$LOG_DIR/evaluate.log" 2>&1

cat > "$RESULT_DIR/eval_status.json" <<JSON
{
  "status": "completed",
  "completed_at": "$(date -Is)",
  "work_dir": "$WORK_DIR",
  "eval_file": "$EVAL_FILE",
  "seed": $SEED,
  "models": ["${MODEL_NAMES[*]}"]
}
JSON

echo "[done] results written to $RESULT_DIR $(date -Is)"
