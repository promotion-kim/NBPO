#!/usr/bin/env bash
# Score one existing, stability-gated P10 Stage-1 generation on the already-open P8 panel.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 ARM GPU" >&2; exit 2; }

ARM=$1
GPU=$2
PROJECT=${PROJECT:-/NHNHOME/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/AIPR/sjkim/venv_clean}
ROOT=${ROOT:-/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1}
E=${E:-$PROJECT/results/p10_saferlhf_training_seed43_20260718}
OUT=$E/stage1_eval_p8_locked_panel
GEN=$OUT/generations/$ARM/output_43.json
SCORES=$OUT/scores_individual/$ARM
INPUT=$SCORES/input.jsonl
REWARD=$ROOT/flagship_20260712/cache/huggingface/models--PKU-Alignment--beaver-7b-v1.0-reward/snapshots/375cd6a9f0d7e339d2199b05ba129a4a8906596d
COST=$ROOT/flagship_20260712/cache/huggingface/models--PKU-Alignment--beaver-7b-v1.0-cost/snapshots/c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28

[[ -s "$GEN" ]] || { echo "missing generation: $GEN" >&2; exit 1; }
mkdir -p "$SCORES"
"$VENV/bin/python" "$PROJECT/analysis/p10_saferlhf_training_seed43_20260718/make_single_policy_score_input.py" \
  --generation "$GEN" --model "p10_stage1_${ARM}_s43" --output "$INPUT" --expected-records 1000 \
  > "$SCORES/input_audit.json"

for _ in 1 2 3; do
  pids=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
  printf '%s gpu=%s compute_pids=%s\n' "$(date -Is)" "$GPU" "${pids:-none}"
  [[ -z ${pids//[[:space:]]/} ]] || { echo "GPU $GPU is not idle" >&2; exit 1; }
  sleep 3
done

run_score() {
  local objective=$1 scorer=$2 model=$3 revision=$4 out=$5
  [[ -s "$out" ]] && return
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PROJECT TOKENIZERS_PARALLELISM=false "$VENV/bin/python" "$scorer" \
    --model_name "$model" --model_revision "$revision" --input_file "$INPUT" --output_file "$out" \
    --batch_size 16 --sample_batch_size 1 --max_seq_length 4096 \
    > "$SCORES/${objective}_$(hostname).log" 2>&1
}
run_score helpfulness "$PROJECT/on_policy_data_gen/rm_beaver_reward.py" "$REWARD" 375cd6a9f0d7e339d2199b05ba129a4a8906596d "$SCORES/helpfulness.jsonl"
run_score harmlessness "$PROJECT/on_policy_data_gen/rm_beaver_cost.py" "$COST" c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28 "$SCORES/harmlessness.jsonl"
