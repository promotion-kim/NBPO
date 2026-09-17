#!/usr/bin/env bash
# Score ONE trained candidate on the held-out pairs and evaluate the Section-7 gate.
#
# The candidate's log-probabilities come from mnpo_scripts.precompute, not from a
# second implementation, so the tokenizer, chat template, response mask,
# truncation rule and the sequence-SUM reduction are the ones training used.
# `--ref_model` is the CANDIDATE (its logps land in the reference_* columns) and
# `--history_paths` is the PARENT, so history0_* is pi_t and h = cand - pi_t.
set -euo pipefail

CKPT="$1"          # candidate checkpoint directory
ETA="$2"           # the eta this candidate was trained at
LABEL="$3"         # arm label, e.g. eta0p3
GPU="${4:-0}"

S=/work/iclr27_table1_v2/smoke/train2
PARENT=/work/models/bases/Llama-3.1-8B-Instruct
OUT=$S/scored/$LABEL

mkdir -p "$OUT"
cd /work/iclr27_table1_v2/code

CUDA_VISIBLE_DEVICES="$GPU" HF_HOME=/root/hf_cache python3 -u -m mnpo_scripts.precompute \
  --model_name_or_path "$PARENT" \
  --ref_model "$CKPT" \
  --history_paths "$PARENT" \
  --train_dir $S/pairs_nbpo/pairs_test.jsonl \
  --output_dir "$OUT/precomputed" \
  --solver_artifact_path $S/solver_nbpo/solution.json \
  --logp_reduction sum --truncation_mode keep_end --ronpo_target_mode none \
  --apply_chat_template true --max_length 2048 --max_prompt_length 1024 \
  --per_device_train_batch_size 2

python3 scripts/experiments/iclr2027_table1_v2/eval_neural_realization.py \
  --candidate-precomputed "$OUT/precomputed" --split train \
  --heldout-split $S/pairs_nbpo/heldout_split.json \
  --eta "$ETA" --method nbpo --seed 42 --label "$LABEL" \
  --expected-solver-hash b9170ba8a2622873e161a0f18fb8d29799350ccafe92fb0df9c81f205886958c \
  --out "$OUT/gate.json"
