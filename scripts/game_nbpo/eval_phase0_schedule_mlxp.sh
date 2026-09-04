#!/usr/bin/env bash
set -euo pipefail
ROOT=/work/campaign_20260824/phase0_schedule
CODE=/work/mnpo_code
mkdir -p "$ROOT/eval_configs" "$ROOT/eval_logs"
export PYTHONPATH=/work/pylibs2:$CODE
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false MNPO_EVAL_ONLY=1
export WANDB_MODE=disabled

run_one() {
  local gpu=$1
  local steps=$2
  local cfg="$ROOT/eval_configs/step${steps}.yaml"
  cp "$ROOT/configs/step${steps}.yaml" "$cfg"
  sed -i \
    -e "s|^model_name_or_path:.*|model_name_or_path: $ROOT/step${steps}|" \
    -e "s|^output_dir:.*|output_dir: $ROOT/eval_step${steps}|" \
    -e '/^report_to:/,/^- wandb$/c\report_to: []' \
    "$cfg"
  CUDA_VISIBLE_DEVICES=$gpu python3 -m accelerate.commands.launch \
    --config_file "$CODE/accelerate_configs/single_gpu.yaml" --num_processes=1 \
    --main_process_port=$((18200 + gpu)) -m mnpo_scripts.run_mnpo "$cfg" \
    >"$ROOT/eval_logs/step${steps}.log" 2>&1
}

run_one 0 20 & p0=$!
run_one 1 50 & p1=$!
run_one 2 100 & p2=$!
run_one 3 200 & p3=$!
rc=0
for p in "$p0" "$p1" "$p2" "$p3"; do wait "$p" || rc=1; done
exit "$rc"
