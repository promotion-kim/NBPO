#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/venv_clean}
EXP=$PROJECT/results/p2_8b_hh_multiobjective_20260717
ROOT=/NHNHOME/26msit001_A/BASE/AIPR/sjkim/revision_qwen3_8b/full_iter1
BASE=$ROOT/base_objective_screen/hf_ipv4/llama31
DATA=$EXP/train_pool/precompute/shared_targets

[[ -s "$EXP/train_pool/DATA_COMPLETE" ]]
[[ -d "$DATA" ]]
source "$VENV/bin/activate"
export PYTHONPATH=$PROJECT
export TORCH_CUDNN_SDPA_ENABLED=0
mkdir -p "$EXP/logs/train" "$EXP/wandb"

: >"$EXP/logs/train/prelaunch_gpu_samples.txt"
for sample in 1 2 3; do
  date -Is >>"$EXP/logs/train/prelaunch_gpu_samples.txt"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader >>"$EXP/logs/train/prelaunch_gpu_samples.txt"
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader)
  printf '%s\n' "$apps" >>"$EXP/logs/train/prelaunch_gpu_samples.txt"
  [[ -z "$apps" ]] || { echo "compute process present before training; fail closed" >&2; exit 8; }
  [[ "$sample" -eq 3 ]] || sleep 2
done

python "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/train_core_arms.py" \
  --project "$PROJECT" --venv "$VENV" --root "$EXP" --model "$BASE" --dataset "$DATA" \
  --stage smoke --wandb-env "$PROJECT/.secrets/wandb.env" --gpus 0,1,2,3 \
  >"$EXP/logs/train/smoke_orchestrator.log" 2>&1

python "$PROJECT/analysis/p2_8b_hh_multiobjective_20260717/train_core_arms.py" \
  --project "$PROJECT" --venv "$VENV" --root "$EXP" --model "$BASE" --dataset "$DATA" \
  --stage full --wandb-env "$PROJECT/.secrets/wandb.env" --gpus 0,1,2,3 \
  >"$EXP/logs/train/full_orchestrator.log" 2>&1

date -Is >"$EXP/train/CORE_TRAIN_COMPLETE"
