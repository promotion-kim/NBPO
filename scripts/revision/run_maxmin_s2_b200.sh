#!/usr/bin/env bash
# MaxMin-RLHF stage-2 arm on a B200 host. Waits for the INPO stage-2 pool
# (merged scores) from run_avg_s2_b200.sh METHOD=inpo, relabels pairs with the
# multiplicative-weights mixture (worst objective upweighted), then precomputes
# and trains with the identical INPO candidate-d recipe, so the only difference
# from INPO-avg stage 2 is the oracle weighting.
#   GPU=0 bash scripts/revision/run_maxmin_s2_b200.sh
set -euo pipefail

GPU="${GPU:?set GPU index}"
SJ=/NHNHOME/AIPR/sjkim
PROJECT_ROOT="$SJ/MNPO_rev_20260720"
RUN_ROOT="$SJ/avg_s2_20260720"
PARENT="$SJ/baseline_repair_1p5b_20260714/candidates/repair1p5b_inpo_d_s42"
PAIR_SRC="$RUN_ROOT/work/inpo/baselines/inpo_avg/iter2/pairs_avg_oracle"
OUT_DIR="$RUN_ROOT/maxmin"
PY="$SJ/venv_clean/bin/python"
# INPO-avg stage-1 objective-level normalized reward (paper stage-1 table)
PERF="skywork=0.5962,athene=0.6768,armo=0.6208"

mkdir -p "$OUT_DIR" "$RUN_ROOT/logs"
export PYTHONPATH="$PROJECT_ROOT"
export HF_HOME="$SJ/baseline_repair_1p5b_20260714/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export WANDB_DIR="$RUN_ROOT/wandb"
export WANDB_ENTITY=promotion-kim WANDB_PROJECT=mnpo WANDB_MODE=online
export MNPO_DISABLE_APEX=1 TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES="$GPU"

echo "[maxmin] waiting for INPO stage-2 merged scores at $PAIR_SRC"
while [[ ! -s "$PAIR_SRC/train_merged_scores.jsonl" || ! -s "$PAIR_SRC/test_merged_scores.jsonl" ]]; do
  sleep 300
done
sleep 60

"$PY" "$PROJECT_ROOT/scripts/revision/build_maxmin_avg_pairs.py" \
  --train_merged "$PAIR_SRC/train_merged_scores.jsonl" \
  --test_merged "$PAIR_SRC/test_merged_scores.jsonl" \
  --output_dir "$OUT_DIR/pairs" \
  --objective_perf "$PERF" --mw_eta 20 --preference_scale 8.0

"$PY" -m accelerate.commands.launch --num_processes=1 -m mnpo_scripts.precompute \
  --model_name_or_path "$PARENT" \
  --ref_model Qwen/Qwen2.5-1.5B-Instruct \
  --history_paths "$PARENT" \
  --train_dir "$OUT_DIR/pairs/train_maxmin_oracle.jsonl" \
  --test_dir "$OUT_DIR/pairs/test_maxmin_oracle.jsonl" \
  --output_dir "$OUT_DIR/precomputed" \
  --per_device_train_batch_size 2 --max_length 2048 --max_prompt_length 1800 \
  --apply_chat_template true --auto_insert_empty_system_msg false \
  --ronpo_target_mode none --sanity_check False \
  2>&1 | tee "$RUN_ROOT/logs/maxmin_precompute_$(date +%m%d_%H%M).log"

RUN_NAME="qwen2.5-1.5b-instruct_inpo_maxmin_online_multiobj_stage_2"
"$PY" -m accelerate.commands.launch \
  --config_file "$PROJECT_ROOT/accelerate_configs/single_gpu.yaml" \
  --num_processes=1 --main_process_port="$((29700 + GPU))" \
  -m mnpo_scripts.run_mnpo \
  "$PROJECT_ROOT/training_configs/inpo/qwen2.5-1.5b-instruct-inpo-avg-multiobj-iter1.yaml" \
  --model_name_or_path="$PARENT" \
  --dataset_mixer="$OUT_DIR/precomputed:1.0" \
  --output_dir="$RUN_ROOT/out/qwen2.5-1.5b-instruct_inpo_maxmin_online_multiobj_stage_2" \
  --run_name="$RUN_NAME" \
  --loss_type=inpo --eta=0.01 --ratio=0.3333 --beta=10 \
  --max_history_t=1 --history_weights=1.0 \
  --reference_anchor_weight=0.05 --preference_sft_weight=0.005 \
  --learning_rate=2.5e-7 \
  --generate_during_eval=false \
  --eval_steps=400 --save_steps=400 --save_total_limit=1 --logging_steps=5 \
  2>&1 | tee "$RUN_ROOT/logs/maxmin_train_$(date +%m%d_%H%M).log"
