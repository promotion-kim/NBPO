#!/usr/bin/env bash
# Reuse the audited parent-plus-base pool builder in a separate continuation directory.
set -euo pipefail
[[ $# -ge 6 && $# -le 7 ]] || { echo "usage: $0 PROJECT EXPERIMENT STAGE ARM PARENT GPU_A [GPU_B]" >&2; exit 2; }
PROJECT=$1
EXP=$2
STAGE=$3
ARM=$4
PARENT=$5
GPU_A=$6
GPU_B=${7:-$GPU_A}
[[ "$STAGE" == stage3 || "$STAGE" == stage4 ]] || { echo "unsupported stage: $STAGE" >&2; exit 2; }
P5_PREPARE=$PROJECT/analysis/p5_8b_robust_stage1_stage2_20260717/prepare_stage2_pool.sh
[[ -d "$PARENT" ]] || { echo "missing parent model: $PARENT" >&2; exit 1; }
mkdir -p "$EXP/$STAGE"
if [[ -e "$EXP/stage2" && ! -L "$EXP/stage2" ]]; then
  echo "refusing non-link $EXP/stage2" >&2
  exit 1
fi
if [[ ! -e "$EXP/stage2" ]]; then
  # Multiple arm-local early launchers may initialize the common alias at the
  # same time. Exactly one creates it; the others accept only the same symlink.
  ln -s "$STAGE" "$EXP/stage2" 2>/dev/null || [[ -L "$EXP/stage2" ]]
fi
POOL=$EXP/$STAGE/$ARM/pool
mkdir -p "$POOL"
printf 'stage=%s\narm=%s\nparent=%s\n' "$STAGE" "$ARM" "$PARENT" > "$POOL/CONTINUATION_POOL_PROVENANCE.txt"
P4="$PROJECT/results/p4_8b_saferlhf_table4_20260717" VLLM_GPU_MEMORY_UTILIZATION=0.55 \
  bash "$P5_PREPARE" "$EXP" "$ARM" "$PARENT" "$GPU_A" "$GPU_B"
test -f "$POOL/PREPARED"
