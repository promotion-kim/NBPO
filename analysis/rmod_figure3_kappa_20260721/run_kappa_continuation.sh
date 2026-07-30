#!/usr/bin/env bash
# Continue an already-trained SafeRLHF RONPO-OS Stage-1 model through Stage 4.
set -euo pipefail

[[ $# -ge 6 && $# -le 7 ]] || { echo "usage: $0 PROJECT ROOT LABEL KAPPA STAGE1_MODEL MAIN_GPU [HELPER_GPU]" >&2; exit 2; }
PROJECT=$1
ROOT=$2
LABEL=$3
KAPPA=$4
STAGE1=$5
GPU=$6
HELPER_GPU=${7:-$GPU}
VENV=$PROJECT/../venv_clean
ARM=ronpo_os_${LABEL}
TARGET=target_os_k${LABEL#k}
P5=$PROJECT/analysis/p5_8b_robust_stage1_stage2_20260717
P10=$PROJECT/analysis/p10_saferlhf_training_seed43_20260718
mkdir -p "$ROOT/$LABEL"
cp "$ROOT/run_lock.json" "$ROOT/$LABEL/run_lock.json"

passed_gate() {
  python - "$1" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
raise SystemExit(0 if x.get("passed") is True and x.get("status") == "passed" else 1)
PY
}

# A new large-kappa Stage-1 checkpoint must pass the unchanged full-panel
# reward-blind gate before it can become a continuation parent.
JOINT=$ROOT/eval_joint
mkdir -p "$JOINT/generations/base" "$JOINT/gates"
ln -sfn "$PROJECT/results/p8_stage4_fresh_default_test_20260718/stage4_eval/generations/base/output_42.json" \
  "$JOINT/generations/base/output_42.json"
if [[ ! -s "$JOINT/gates/${LABEL}_stage1.json" ]]; then
  bash "$PROJECT/analysis/p11_saferlhf_stage_curve_20260718/decode_and_gate.sh" \
    "$PROJECT" "$JOINT" "${LABEL}_stage1" "$STAGE1" "$GPU"
fi
passed_gate "$JOINT/gates/${LABEL}_stage1.json"

prepare_with_helper() {
  local exp=$1 stage=$2 parent=$3
  flock -x "/tmp/figure3_kappa_${GPU}_${HELPER_GPU}.lock" bash -c '
    set -euo pipefail
    export RONPO_KAPPA="$1"
    if [[ "$2" == stage2 ]]; then
      bash "$3/analysis/p5_8b_robust_stage1_stage2_20260717/prepare_stage2_pool.sh" "$4" "$5" "$6" "$7" "$8"
    else
      bash "$3/analysis/p10_saferlhf_training_seed43_20260718/prepare_continuation_pool.sh" "$3" "$4" "$2" "$5" "$6" "$7" "$8"
    fi
  ' _ "$KAPPA" "$stage" "$PROJECT" "$exp" "$ARM" "$parent" "$GPU" "$HELPER_GPU"
}

run_gate() {
  local exp=$1 stage=$2
  local gate=$exp/${stage}_stability_p8_locked_panel/gates/$ARM.json
  if [[ ! -s "$gate" ]]; then
    TRAIN_SEED=42 RUN_PREFIX=fig3-${LABEL} \
      bash "$P10/decode_and_gate_continuation.sh" "$PROJECT" "$exp" "$stage" "$ARM" "$GPU"
  fi
  passed_gate "$gate"
}

S2=$ROOT/$LABEL/stage2_run
mkdir -p "$S2"
cp "$ROOT/run_lock.json" "$S2/run_lock.json"
if [[ ! -s "$S2/stage2/$ARM/pool/PREPARED" ]]; then
  prepare_with_helper "$S2" stage2 "$STAGE1"
fi
if [[ ! -s "$S2/stage2/$ARM/train/smoke/job_status.json" ]]; then
  "$VENV/bin/python" "$P5/train_stage2_arm.py" --project "$PROJECT" --venv "$VENV" --experiment "$S2" \
    --arm "$ARM" --parent-model "$STAGE1" --loss-type ronpo --target-column "$TARGET" --gpu "$GPU" --stage smoke
fi
if [[ ! -s "$S2/stage2/$ARM/train/full/job_status.json" ]]; then
  "$VENV/bin/python" "$P5/train_stage2_arm.py" --project "$PROJECT" --venv "$VENV" --experiment "$S2" \
    --arm "$ARM" --parent-model "$STAGE1" --loss-type ronpo --target-column "$TARGET" --gpu "$GPU" --stage full
fi
run_gate "$S2" stage2
PARENT=$S2/stage2/$ARM/train/full

for STAGE in stage3 stage4; do
  EXP=$ROOT/$LABEL/${STAGE}_run
  mkdir -p "$EXP"
  cp "$ROOT/run_lock.json" "$EXP/continuation_lock.json"
  if [[ ! -s "$EXP/$STAGE/$ARM/pool/PREPARED" ]]; then
    prepare_with_helper "$EXP" "$STAGE" "$PARENT"
  fi
  if [[ ! -s "$EXP/$STAGE/$ARM/train/smoke/job_status.json" ]]; then
    TRAIN_SEED=42 RUN_PREFIX=fig3-${LABEL} \
      "$VENV/bin/python" "$P10/train_continuation_arm.py" --project "$PROJECT" --venv "$VENV" \
        --experiment "$EXP" --continuation-stage "$STAGE" --arm "$ARM" --parent-model "$PARENT" \
        --loss-type ronpo --target-column "$TARGET" --gpu "$GPU" --seed 42 --run-prefix fig3-${LABEL} --run-stage smoke
  fi
  if [[ ! -s "$EXP/$STAGE/$ARM/train/full/job_status.json" ]]; then
    TRAIN_SEED=42 RUN_PREFIX=fig3-${LABEL} \
      "$VENV/bin/python" "$P10/train_continuation_arm.py" --project "$PROJECT" --venv "$VENV" \
        --experiment "$EXP" --continuation-stage "$STAGE" --arm "$ARM" --parent-model "$PARENT" \
        --loss-type ronpo --target-column "$TARGET" --gpu "$GPU" --seed 42 --run-prefix fig3-${LABEL} --run-stage full
  fi
  run_gate "$EXP" "$STAGE"
  PARENT=$EXP/$STAGE/$ARM/train/full
done

date -Is > "$ROOT/$LABEL/COMPLETE"
