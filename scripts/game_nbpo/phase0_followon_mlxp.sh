#!/usr/bin/env bash
set -euo pipefail
ROOT=/work/campaign_20260824/phase0_schedule
CODE=/work/mnpo_code
export PYTHONPATH=/work/pylibs2:$CODE
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

pid=$(cat "$ROOT/step200.retry.pid")
while state=$(ps -o stat= -p "$pid" 2>/dev/null) && [[ "$state" != Z* ]]; do sleep 30; done
test -f "$ROOT/step200/train_results.json"
test "$(find "$ROOT/step200" -maxdepth 1 -name '*.safetensors' | wc -l)" -eq 7

cfg="$ROOT/eval_configs/step200.yaml"
cp "$ROOT/configs/step200.yaml" "$cfg"
sed -i \
  -e "s|^model_name_or_path:.*|model_name_or_path: $ROOT/step200|" \
  -e "s|^output_dir:.*|output_dir: $ROOT/eval_step200|" \
  -e '/^report_to:/,/^- wandb$/c\report_to: []' \
  "$cfg"
CUDA_VISIBLE_DEVICES=3 MNPO_EVAL_ONLY=1 WANDB_MODE=disabled \
  python3 -m accelerate.commands.launch \
  --config_file "$CODE/accelerate_configs/single_gpu.yaml" --num_processes=1 \
  --main_process_port=18203 -m mnpo_scripts.run_mnpo "$cfg" \
  >"$ROOT/eval_logs/step200.log" 2>&1

python3 /work/campaign_20260824/summarize_schedule.py --root "$ROOT" \
  >"$ROOT/schedule_summary.log"
selected=$(python3 -c "import json; print(json.load(open('$ROOT/schedule_summary.json'))['selected_steps'])")
echo "$selected" > "$ROOT/selected_steps.txt"

pool="$ROOT/selected_pool"
rm -rf "$pool"
mkdir -p "$pool"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 /work/oracle/oracle_round_game.py \
  --policy "$ROOT/step$selected" \
  --anchor /work/models/bases/Llama-3.1-8B-Instruct \
  --judge /work/models/xj_judges/qwen3-32b \
  --prompts /work/gtpref/ultrafeedback/pairs_train.jsonl \
  --out "$pool" \
  --anchor-cache /work/game/ultrafeedback_s42_b0.25/anchor_gens.json \
  --objectives instruction_following,truthfulness,honesty,helpfulness \
  --n 1500 --n-policy 4 --n-anchor 4 --cycles-on 0 --tp 4 \
  >"$ROOT/selected_pool.log" 2>&1

python3 /work/oracle/oracle_pairs_game.py --pool "$pool" --out "$pool/pairs" \
  --eps 0.02 --uniform >"$ROOT/selected_pairs.log" 2>&1
python3 /work/game/game_targets.py --pool "$pool" --pairs "$pool/pairs" \
  --beta 0.25 --eps 0.02 \
  --ref-values /work/game/ultrafeedback_s42_b0.25/refvals.json \
  >"$ROOT/selected_targets.log" 2>&1
python3 /work/campaign_20260824/compute_phase0_gate.py \
  --game-targets /work/game/game_targets.py \
  --reference-pool /work/game/ultrafeedback_s42_b0.25/refpool \
  --selected-pool "$pool" \
  --reference-values /work/game/ultrafeedback_s42_b0.25/refvals.json \
  --output "$ROOT/phase0_gate.json" >"$ROOT/phase0_gate.log"
date -Iseconds > "$ROOT/followon.complete"
