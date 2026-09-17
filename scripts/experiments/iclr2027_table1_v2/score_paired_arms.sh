#!/usr/bin/env bash
# Score the paired reference arms against the canonical target.
#
# Both arms are scored by the SAME evaluator, from the frozen code tree, so the
# comparison isolates the training-time reference path rather than also changing
# the ruler. The evaluator's own log-probabilities come from precompute for both
# terms, exactly as every earlier arm was scored.
set -euo pipefail

ARM="$1"; GPU="${2:-0}"
C=/work/iclr27_table1_v2/smoke/canon
R=/work/iclr27_table1_v2/smoke/reffix
PARENT=/work/models/bases/Llama-3.1-8B-Instruct
OUT=$R/scored/$ARM
mkdir -p "$OUT"
cd /work/iclr27_table1_v2/code_frozen_0131
export HF_HOME=/work/hf_cache

CUDA_VISIBLE_DEVICES="$GPU" python3 -u -m mnpo_scripts.precompute \
  --model_name_or_path "$PARENT" --ref_model "$R/arms/$ARM" --history_paths "$PARENT" \
  --train_dir $C/pairs_nbpo/pairs_test.jsonl \
  --output_dir "$OUT/precomputed" \
  --solver_artifact_path $C/solver_nbpo/solution.json \
  --logp_reduction sum --truncation_mode keep_end --ronpo_target_mode none \
  --apply_chat_template true --max_length 2048 --max_prompt_length 1024 \
  --per_device_train_batch_size 2

python3 scripts/experiments/iclr2027_table1_v2/eval_neural_realization.py \
  --candidate-precomputed "$OUT/precomputed" --split train \
  --heldout-split $C/pairs_nbpo/heldout_split.json \
  --eta 1.0 --method nbpo --seed 42 --label "$ARM" \
  --expected-solver-hash b9170ba8a2622873e161a0f18fb8d29799350ccafe92fb0df9c81f205886958c \
  --out "$OUT/gate.json"
