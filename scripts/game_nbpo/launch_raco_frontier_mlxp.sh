#!/usr/bin/env bash
set -euo pipefail

ROOT=/work/campaign_20260824
CODE=/work/repos/RACO
MODEL=/work/models/bases/Llama-3.1-8B-Instruct
TRAIN=$ROOT/data/reddit/train_conflicted.jsonl
VAL=$ROOT/data/reddit/val.jsonl
mkdir -p "$ROOT/raco_ckpts" "$ROOT/logs"

export PYTHONPATH="$CODE/trl:/work/pylibs2"
export HF_HOME=/work/cache/hf_campaign
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=mnpo
export WANDB_ENTITY=promotion-kim
export WANDB_MODE=online

run_arm() {
  local gpu=$1 name=$2 weights=$3 cagrad=$4 clip=$5
  local out="$ROOT/raco_ckpts/$name"
  if [[ -s "$out/model.safetensors" || -s "$out/model-00001-of-"*".safetensors" ]]; then
    echo "[$(date -Iseconds)] SKIP_COMPLETE $name"
    return 0
  fi
  rm -rf "$out"
  mkdir -p "$out"
  echo "[$(date -Iseconds)] START $name gpu=$gpu" | tee "$ROOT/logs/$name.status"
  CUDA_VISIBLE_DEVICES=$gpu python3 "$CODE/trl/scripts/train_raco.py" \
    --mode raco \
    --model_name_or_path "$MODEL" \
    --dataset_path "$TRAIN" \
    --val_dataset_path "$VAL" \
    --output_dir "$out" \
    --max_length 2048 \
    --max_prompt_length 1536 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --learning_rate 7e-7 \
    --beta 0.2 \
    --seed 42 \
    --num_train_epochs 1 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type linear \
    --bf16 True \
    --gradient_checkpointing True \
    --raco True \
    --raco_num_objectives 2 \
    --raco_weights "$weights" \
    --raco_c 0.4 \
    --raco_use_cagrad "$cagrad" \
    --raco_clip_lambda "$clip" \
    --eval_strategy no \
    --logging_steps 5 \
    --report_to wandb \
    --wandb_project mnpo \
    --wandb_entity promotion-kim \
    --run_name "game-nbpo-raco-${name}-s42" \
    >"$ROOT/logs/$name.train.log" 2>&1
  python3 - "$out" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
weights = list(p.glob("*.safetensors"))
if not weights:
    raise SystemExit("missing final safetensors")
(p / "TRAINING_COMPLETE.json").write_text(json.dumps({
    "status": "complete", "weight_files": [x.name for x in weights]
}, indent=2) + "\n")
PY
  echo "[$(date -Iseconds)] COMPLETE $name" | tee -a "$ROOT/logs/$name.status"
}

run_arm 0 raco_equal 0.5,0.5 True True &
p0=$!
run_arm 1 raco_quality 0.8,0.2 True True &
p1=$!
run_arm 2 raco_verbosity 0.2,0.8 True True &
p2=$!
run_arm 3 dpolw_equal 0.5,0.5 False False &
p3=$!

rc=0
for p in "$p0" "$p1" "$p2" "$p3"; do wait "$p" || rc=1; done
date -Iseconds > "$ROOT/raco_frontier_finished_at.txt"
exit "$rc"
