#!/usr/bin/env bash
# Reuse the audited Stage-2 pool builder through a private P7 stage2->stage3 link.
# The implementation remains byte-identical apart from the stage path and split salt.
set -euo pipefail
if [[ $# -ne 5 ]]; then echo "usage: $0 EXPERIMENT ARM PARENT GPU_A GPU_B" >&2; exit 2; fi
EXP=$1
ARM=$2
PARENT=$3
GPU_A=$4
GPU_B=$5
PROJECT=${PROJECT:-/NHNHOME/AIPR/sjkim/MNPO_rev_20260710}
P5_PREPARE=${P5_PREPARE:-$PROJECT/analysis/p5_8b_robust_stage1_stage2_20260717/prepare_stage2_pool.sh}
[[ -d "$EXP/stage3" ]] || mkdir -p "$EXP/stage3"
if [[ -e "$EXP/stage2" && ! -L "$EXP/stage2" ]]; then echo "refusing non-link $EXP/stage2" >&2; exit 1; fi
if [[ ! -e "$EXP/stage2" ]]; then ln -s stage3 "$EXP/stage2"; fi
P4=${P4:-$PROJECT/results/p4_8b_saferlhf_table4_20260717} EXP=$EXP bash "$P5_PREPARE" "$EXP" "$ARM" "$PARENT" "$GPU_A" "$GPU_B"
POOL=$EXP/stage3/$ARM/pool
printf '%s\n' "stage3_pool_from_parent=$PARENT" > "$POOL/STAGE3_POOL_PROVENANCE.txt"
