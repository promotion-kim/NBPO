#!/usr/bin/env bash
# Score a trained arm on the FIXED TRAIN pairs, with the same evaluator contract
# used for held-out.
#
# Every number reported so far is a held-out endpoint, so "the arm failed" could
# not be separated from "the arm never fit its own training data". This scores
# the final checkpoint on the training rows themselves -- not the online
# minibatch loss, which is an average over a moving policy and is not comparable
# to a final-checkpoint evaluation.
set -euo pipefail
ARM="$1"; GPU="${2:-0}"; PAIRS="${3:-/work/iclr27_table1_v2/smoke/canon/pairs_nbpo/pairs_train.jsonl}"
ARMDIR="${4:-/work/iclr27_table1_v2/smoke/reffix/arms/$ARM}"
C=/work/iclr27_table1_v2/smoke/canon
OUT=/work/iclr27_table1_v2/smoke/trainfit/$ARM
PARENT=/work/models/bases/Llama-3.1-8B-Instruct
mkdir -p "$OUT"
cd /work/iclr27_table1_v2/code_frozen_0131
export HF_HOME=/work/hf_cache

CUDA_VISIBLE_DEVICES="$GPU" python3 -u -m mnpo_scripts.precompute \
  --model_name_or_path "$PARENT" --ref_model "$ARMDIR" --history_paths "$PARENT" \
  --train_dir "$PAIRS" --output_dir "$OUT/precomputed" \
  --solver_artifact_path $C/solver_nbpo/solution.json \
  --logp_reduction sum --truncation_mode keep_end --ronpo_target_mode none \
  --apply_chat_template true --max_length 2048 --max_prompt_length 1024 \
  --per_device_train_batch_size 2
