#!/usr/bin/env bash
# Route Hugging Face and related ML caches away from the home directory.
# Source this file from run scripts before importing transformers/datasets/vLLM.

EXT_CACHE_ROOT="${EXT_CACHE_ROOT:-/ext_hdd/sjkim}"

if [[ "${RESPECT_EXISTING_CACHE:-0}" == "1" ]]; then
  export HF_HOME="${HF_HOME:-$EXT_CACHE_ROOT/huggingface}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HUGGINGFACE_HUB_CACHE}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$EXT_CACHE_ROOT/xdg_cache}"
  export TORCH_HOME="${TORCH_HOME:-$EXT_CACHE_ROOT/torch}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$EXT_CACHE_ROOT/triton_cache}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$EXT_CACHE_ROOT/pip_cache}"
  export WANDB_DIR="${WANDB_DIR:-$EXT_CACHE_ROOT/mnpo/wandb}"
else
  export HF_HOME="$EXT_CACHE_ROOT/huggingface"
  export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
  export HF_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
  export HF_DATASETS_CACHE="$HF_HOME/datasets"
  export TRANSFORMERS_CACHE="$HF_HOME/transformers"

  export XDG_CACHE_HOME="$EXT_CACHE_ROOT/xdg_cache"
  export TORCH_HOME="$EXT_CACHE_ROOT/torch"
  export TRITON_CACHE_DIR="$EXT_CACHE_ROOT/triton_cache"
  export PIP_CACHE_DIR="$EXT_CACHE_ROOT/pip_cache"
  export WANDB_DIR="$EXT_CACHE_ROOT/mnpo/wandb"
fi

# Backward-compatible cache_dir used by this repo's RM/precompute scripts.
if [[ "${RESPECT_EXISTING_CACHE:-0}" == "1" ]]; then
  export CACHE_DIR="${CACHE_DIR:-$HUGGINGFACE_HUB_CACHE}"
else
  export CACHE_DIR="$HUGGINGFACE_HUB_CACHE"
fi

mkdir -p \
  "$HF_HOME" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$XDG_CACHE_HOME" \
  "$TORCH_HOME" \
  "$TRITON_CACHE_DIR" \
  "$PIP_CACHE_DIR" \
  "$WANDB_DIR"
