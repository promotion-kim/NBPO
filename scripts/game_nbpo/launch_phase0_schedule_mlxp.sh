#!/usr/bin/env bash
set -euo pipefail
ROOT=/work/campaign_20260824/phase0_schedule
CODE=/work/mnpo_code
TEMPLATE=/work/campaign_20260824/phase0_schedule_template.yaml
mkdir -p "$ROOT/configs" "$ROOT/logs"
export PYTHONPATH=/work/pylibs2:$CODE
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=online WANDB_ENTITY=promotion-kim WANDB_PROJECT=mnpo

run_one() {
  local gpu=$1
  local steps=$2
  local cfg="$ROOT/configs/step${steps}.yaml"
  sed "s/__STEPS__/${steps}/g" "$TEMPLATE" > "$cfg"
  CUDA_VISIBLE_DEVICES=$gpu python3 -m accelerate.commands.launch \
    --config_file "$CODE/accelerate_configs/single_gpu.yaml" --num_processes=1 \
    --main_process_port=$((18100 + gpu)) -m mnpo_scripts.run_mnpo "$cfg" \
    >"$ROOT/logs/step${steps}.log" 2>&1
  date -Iseconds > "$ROOT/step${steps}.complete"
}

run_one 0 20 & p0=$!
run_one 1 50 & p1=$!
run_one 2 100 & p2=$!
run_one 3 200 & p3=$!
rc=0
for p in "$p0" "$p1" "$p2" "$p3"; do wait "$p" || rc=1; done
date -Iseconds > "$ROOT/finished_at.txt"
exit "$rc"
