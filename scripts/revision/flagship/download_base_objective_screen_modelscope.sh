#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/venv_clean}
ROOT=${ROOT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/revision_qwen3_8b/full_iter1/base_objective_screen/modelscope}
OUT=$PROJECT/results/p1_8b_base_objective_screen_20260716/downloads
mkdir -p "$ROOT" "$OUT"
source "$VENV/bin/activate"

download_one() {
  local name=$1 repo=$2
  local target=$ROOT/$name
  mkdir -p "$target"
  printf '%s\t%s\t%s\tmodelscope_start\n' "$(date -Is)" "$name" "$repo" >>"$OUT/modelscope.log"
  if ms download "$repo" --repo-type model --revision master --local-dir "$target" --max-workers 8 \
      --exclude 'original/*' '*.pth' '*.gguf' '*.bin' 'consolidated.safetensors' \
      >"$OUT/$name.modelscope.log" 2>&1; then
    [[ -f "$target/config.json" ]] || { printf '%s\t%s\tmissing_config\n' "$(date -Is)" "$name" >>"$OUT/modelscope.log"; return 1; }
    printf '%s\n' "$target" >"$OUT/$name.path"
    printf '%s\t%s\tmodelscope_master\tcomplete\n' "$name" "$repo" >"$OUT/$name.status"
    printf '%s\t%s\t%s\tmodelscope_complete\n' "$(date -Is)" "$name" "$repo" >>"$OUT/modelscope.log"
  else
    code=$?; printf '%s\t%s\t%s\tmodelscope_failed_%s\n' "$(date -Is)" "$name" "$repo" "$code" >>"$OUT/modelscope.log"; return "$code"
  fi
}
export -f download_one
export ROOT OUT

download_one llama31 LLM-Research/Meta-Llama-3.1-8B-Instruct & p0=$!
download_one qwen25 Qwen/Qwen2.5-7B-Instruct & p1=$!
download_one mistral7 LLM-Research/Mistral-7B-Instruct-v0.3 & p2=$!
download_one wildguard allenai/wildguard & p3=$!
wait "$p0" "$p1" "$p2" "$p3"
download_one zephyr modelscope/zephyr-7b-beta
date -Is >"$OUT/MODELSCOPE_COMPLETE"
