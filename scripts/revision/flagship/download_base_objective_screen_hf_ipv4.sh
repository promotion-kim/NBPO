#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/venv_clean}
ROOT=${ROOT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/revision_qwen3_8b/full_iter1/base_objective_screen/hf_ipv4}
OUT=$PROJECT/results/p1_8b_base_objective_screen_20260716/downloads
DOWNLOADER=$PROJECT/analysis/qwen3_8b_base_objective_screen_20260716/download_hf_ipv4.py
mkdir -p "$ROOT" "$OUT/metadata"
source "$VENV/bin/activate"
export HF_HUB_DISABLE_XET=1

download_one() {
  local name=$1 repo=$2
  local target=$ROOT/$name
  printf '%s\t%s\t%s\thf_ipv4_start\n' "$(date -Is)" "$name" "$repo" >>"$OUT/hf_ipv4.log"
  if python "$DOWNLOADER" --repo "$repo" --local-dir "$target" --metadata "$OUT/metadata/$name.json" \
      >"$OUT/$name.hf_ipv4.log" 2>&1; then
    revision=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["resolved_revision"])' "$OUT/metadata/$name.json")
    printf '%s\n' "$target" >"$OUT/$name.path"
    printf '%s\t%s\t%s\tcomplete\n' "$name" "$repo" "$revision" >"$OUT/$name.status"
    printf '%s\t%s\t%s\thf_ipv4_complete\n' "$(date -Is)" "$name" "$repo" >>"$OUT/hf_ipv4.log"
  else
    code=$?; printf '%s\t%s\t%s\thf_ipv4_failed_%s\n' "$(date -Is)" "$name" "$repo" "$code" >>"$OUT/hf_ipv4.log"; return "$code"
  fi
}
export -f download_one
export ROOT OUT DOWNLOADER

download_one llama31 meta-llama/Llama-3.1-8B-Instruct & p0=$!
download_one qwen25 Qwen/Qwen2.5-7B-Instruct & p1=$!
download_one mistral7 mistralai/Mistral-7B-Instruct-v0.3 & p2=$!
download_one wildguard allenai/wildguard & p3=$!
wait "$p0" "$p1" "$p2" "$p3"
download_one zephyr HuggingFaceH4/zephyr-7b-beta
date -Is >"$OUT/HF_IPV4_COMPLETE"
