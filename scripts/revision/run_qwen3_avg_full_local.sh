#!/usr/bin/env bash
set -uo pipefail

METHOD="${1:?method required: dpo or ipo}"
LOSS_TYPE="${2:?loss type required: dpo or ipo}"
BETAS_CSV="${3:?comma-separated beta values required}"
CUDA_DEVICES="${4:?CUDA device list required, e.g. 0,1}"
NUM_PROCESSES="${5:?num processes required}"
PER_DEVICE_BS="${6:?per-device train batch size required}"
GRAD_ACCUM="${7:?gradient accumulation steps required}"
HOST_TAG="${8:?host tag required}"

SEED="${SEED:-42}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/sjkim/MNPO}"
EXP_ROOT="${EXP_ROOT:-/ext_hdd/sjkim/mnpo/revision_qwen3_8b/full_iter1}"
PREF_DIR="${PREF_DIR:-/home/sjkim/mnpo_shared/revision_qwen3_8b/full_iter1/precomputed/avg_oracle}"
MODEL_ID="${MODEL_ID:-/ext_hdd/sjkim/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
CACHE_ROOT="${CACHE_ROOT:-/ext_hdd/sjkim/huggingface}"
CONDA_PYTHON="${CONDA_PYTHON:-/home/sjkim/anaconda3/envs/ronpo-rev/bin/python}"
ACCEL_CONFIG="${ACCEL_CONFIG:-$PROJECT_ROOT/accelerate_configs/deepspeed_zero3_port29501.yaml}"
RUN_GROUP="${RUN_GROUP:-ronpo-revision-qwen3-8b-avg-baselines}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
SMOKE_STEPS="${SMOKE_STEPS:-0}"

LOG_DIR="$EXP_ROOT/logs"
PATCH_DIR="$EXP_ROOT/runtime/sitecustomize_${METHOD}_${HOST_TAG}"
FAKE_CUDA_HOME="$EXP_ROOT/runtime/fake_cuda_${METHOD}_${HOST_TAG}"

mkdir -p \
  "$LOG_DIR" "$PATCH_DIR" "$FAKE_CUDA_HOME/bin" "$EXP_ROOT/wandb" \
  "$CACHE_ROOT"/{hub,datasets,transformers,torch,triton}

if [[ "$METHOD" != "dpo" && "$METHOD" != "ipo" ]]; then
  echo "[fatal] METHOD must be dpo or ipo, got $METHOD" >&2
  exit 2
fi
if [[ "$LOSS_TYPE" != "dpo" && "$LOSS_TYPE" != "ipo" ]]; then
  echo "[fatal] LOSS_TYPE must be dpo or ipo, got $LOSS_TYPE" >&2
  exit 2
fi
if [[ ! -f "$PROJECT_ROOT/mnpo_scripts/run_mnpo.py" ]]; then
  echo "[fatal] missing project root: $PROJECT_ROOT" >&2
  exit 2
fi
if [[ ! -d "$PREF_DIR/train" || ! -d "$PREF_DIR/test" ]]; then
  echo "[fatal] missing precomputed avg_oracle train/test under $PREF_DIR" >&2
  exit 2
fi
if [[ ! -f "$MODEL_ID/config.json" || ! -f "$MODEL_ID/model.safetensors.index.json" ]]; then
  echo "[fatal] incomplete Qwen3-8B snapshot at $MODEL_ID" >&2
  exit 2
fi
if [[ ! -x "$CONDA_PYTHON" ]]; then
  echo "[fatal] missing python executable: $CONDA_PYTHON" >&2
  exit 2
fi

cat > "$PATCH_DIR/sitecustomize.py" <<'PY'
import importlib.util
import inspect

_orig_find_spec = importlib.util.find_spec

def _find_spec_without_broken_apex(name, package=None):
    hidden_prefixes = (
        "apex",
        "flash_attn",
        "flash_attn_2_cuda",
        "flash_attn_cuda",
        "flash_attn_interface",
    )
    if name in hidden_prefixes or any(name.startswith(prefix + ".") for prefix in hidden_prefixes):
        return None
    return _orig_find_spec(name, package)

importlib.util.find_spec = _find_spec_without_broken_apex

try:
    from accelerate import Accelerator
    _orig_unwrap_model = Accelerator.unwrap_model
    _params = inspect.signature(_orig_unwrap_model).parameters
    if "keep_torch_compile" not in _params:
        def _unwrap_model_compat(self, model, keep_fp32_wrapper=True, keep_torch_compile=True):
            return _orig_unwrap_model(self, model, keep_fp32_wrapper=keep_fp32_wrapper)
        Accelerator.unwrap_model = _unwrap_model_compat
except Exception as exc:
    print(f"[sitecustomize] accelerate compatibility patch skipped: {exc}")
PY

cat > "$FAKE_CUDA_HOME/bin/nvcc" <<'SH'
#!/usr/bin/env bash
echo "nvcc: NVIDIA (R) Cuda compiler driver"
echo "Cuda compilation tools, release 12.1, V12.1.105"
SH
chmod +x "$FAKE_CUDA_HOME/bin/nvcc"

cd "$PROJECT_ROOT"

IFS=',' read -r -a BETAS <<< "$BETAS_CSV"
for beta in "${BETAS[@]}"; do
  beta_id="${beta//./p}"
  output_dir="$EXP_ROOT/train/${METHOD}_avg_beta${beta_id}_s${SEED}_${HOST_TAG}"
  config="$output_dir/config.yaml"
  log_file="$LOG_DIR/train_${METHOD}_avg_beta${beta_id}_s${SEED}_${HOST_TAG}_$(date +%Y%m%d_%H%M%S).log"
  mkdir -p "$output_dir"

  eval_strategy="steps"
  do_eval="true"
  save_strategy="steps"
  max_steps_line=""
  if [[ "$SMOKE_STEPS" != "0" ]]; then
    eval_strategy="no"
    do_eval="false"
    save_strategy="no"
    max_steps_line="max_steps: $SMOKE_STEPS"
  fi

  cat > "$config" <<YAML
model_name_or_path: $MODEL_ID
torch_dtype: null
attn_implementation: $ATTN_IMPLEMENTATION
dataset_mixer:
  $PREF_DIR: 1.0
dataset_splits:
  - train
  - test
preprocessing_num_workers: 12
bf16: true
loss_type: $LOSS_TYPE
dpo_beta: $beta
simpo_beta: 2.5
simpo_gamma: 1.0
eta: 0.0075
ratio: 0.3333
max_history_t: 1
history_weights: [1.0]
beta: 10
learning_rate: 5.0e-7
lr_scheduler_type: cosine
warmup_ratio: 0.1
optim: adamw_torch
weight_decay: 0.0
seed: $SEED
gradient_accumulation_steps: $GRAD_ACCUM
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
num_train_epochs: 1
$max_steps_line
per_device_train_batch_size: $PER_DEVICE_BS
per_device_eval_batch_size: $PER_DEVICE_BS
max_length: 2048
max_prompt_length: 1800
do_eval: $do_eval
eval_strategy: "$eval_strategy"
eval_steps: 500
logging_steps: 10
log_level: info
generate_during_eval: false
metric_for_best_model: eval_loss
load_best_model_at_end: false
save_strategy: "$save_strategy"
save_steps: 1000
save_total_limit: 1
save_only_model: true
save_safetensors: true
push_to_hub: false
report_to:
  - wandb
output_dir: $output_dir
run_name: rev-q3-${METHOD}-b${beta_id}-s${SEED}-${HOST_TAG}
YAML

  {
    echo "[start] method=$METHOD loss=$LOSS_TYPE beta=$beta seed=$SEED host=$(hostname) devices=$CUDA_DEVICES processes=$NUM_PROCESSES per_device_bs=$PER_DEVICE_BS grad_accum=$GRAD_ACCUM $(date -Is)"
    echo "[paths] project=$PROJECT_ROOT pref=$PREF_DIR model=$MODEL_ID output=$output_dir"
    nvidia-smi || true

    export PATH="$(dirname "$CONDA_PYTHON"):$PATH"
    export PYTHONPATH="$PATCH_DIR:$PROJECT_ROOT"
    export HF_HOME="$CACHE_ROOT"
    export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/hub"
    export HF_HUB_CACHE="$CACHE_ROOT/hub"
    export HF_DATASETS_CACHE="$CACHE_ROOT/datasets"
    export TRANSFORMERS_CACHE="$CACHE_ROOT/transformers"
    export TORCH_HOME="$CACHE_ROOT/torch"
    export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
    export WANDB_PROJECT=mnpo
    export WANDB_ENTITY=promotion-kim
    export WANDB_RUN_GROUP="$RUN_GROUP"
    export WANDB_DIR="$EXP_ROOT/wandb"
    export WANDB_MODE="${WANDB_MODE:-online}"
    export MNPO_TOKENIZER_USE_FAST="${MNPO_TOKENIZER_USE_FAST:-0}"
    export ACCELERATE_LOG_LEVEL=info
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=1
    export NCCL_CUMEM_ENABLE=0
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export DS_BUILD_OPS=0
    export DS_SKIP_CUDA_CHECK=1
    export CUDA_HOME="$FAKE_CUDA_HOME"

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$CONDA_PYTHON" -m accelerate.commands.launch \
      --config_file "$ACCEL_CONFIG" \
      --num_processes="$NUM_PROCESSES" \
      -m mnpo_scripts.run_mnpo "$config"
    rc=$?

    "$CONDA_PYTHON" - <<PY
import json, pathlib, time
out = pathlib.Path("$output_dir")
summary = {
    "method": "$METHOD",
    "loss_type": "$LOSS_TYPE",
    "seed": int("$SEED"),
    "beta": float("$beta"),
    "backbone": "Qwen/Qwen3-8B",
    "status": "completed" if int("$rc") == 0 else "failed",
    "returncode": int("$rc"),
    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "dataset": "$PREF_DIR",
    "output_dir": str(out),
    "config": str(out / "config.yaml"),
    "train_metrics_exists": (out / "train_results.json").exists(),
    "trainer_state_exists": (out / "trainer_state.json").exists(),
}
(out / "run_status.json").write_text(json.dumps(summary, indent=2) + "\\n")
print("[run-status]", json.dumps(summary, sort_keys=True))
PY
    exit "$rc"
  } 2>&1 | tee "$log_file"
done
