#!/usr/bin/env bash
# Opportunistically decode/gate existing kappa-stage points on the common P8 panel.
set -euo pipefail
[[ $# -eq 3 ]] || { echo "usage: $0 PROJECT ROOT GPU" >&2; exit 2; }
PROJECT=$1
ROOT=$2
GPU=$3
OUT=$ROOT/eval_joint
P4=$PROJECT/results/p4_8b_saferlhf_table4_20260717
P5=$PROJECT/results/p5_8b_robust_stage1_stage2_20260717
P7=$PROJECT/results/p7_stage3_fresh_default_test_20260717
P8=$PROJECT/results/p8_stage4_fresh_default_test_20260718
DECODER=$PROJECT/analysis/p11_saferlhf_stage_curve_20260718/decode_and_gate.sh
mkdir -p "$OUT/generations/base" "$OUT/logs"
ln -sfn "$P8/stage4_eval/generations/base/output_42.json" "$OUT/generations/base/output_42.json"

run_point() {
  local key=$1 model=$2
  flock -x /tmp/figure3_kappa_gpu3.lock \
    bash "$DECODER" "$PROJECT" "$OUT" "$key" "$model" "$GPU"
}

run_point base /NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1/base_objective_screen/hf_ipv4/llama31
run_point k0p01_stage1 "$P4/train/w2_900steps/ronpo_os_entropy_0p15_diagnostic"
run_point k0p1_stage1 "$P4/train/w1_900steps/ronpo_os_confirmatory"
run_point k0p5_stage1 "$P4/train/w2_900steps/ronpo_os_entropy_0p85_diagnostic"
run_point k0p1_stage2 "$P5/stage2/ronpo_os_stage2/train/full"
run_point k0p1_stage3 "$P7/stage3/ronpo_os_stage3/train/full"
mkdir -p "$OUT/generations/k0p1_stage4"
ln -sfn "$P8/stage4_eval/generations/ronpo_os_stage4/output_42.json" "$OUT/generations/k0p1_stage4/output_42.json"
run_point k0p1_stage4 "$P8/stage4/ronpo_os_stage4/train/full"
date -Is > "$OUT/EXISTING_POINTS_DECODED"
