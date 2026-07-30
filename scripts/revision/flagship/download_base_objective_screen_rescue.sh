#!/usr/bin/env bash
set -euo pipefail
PROJECT=${PROJECT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/venv_clean}
EXP=$PROJECT/results/p1_8b_base_objective_screen_20260716
OUT=$EXP/downloads
MSROOT=/NHNHOME/26msit001_A/BASE/AIPR/sjkim/revision_qwen3_8b/full_iter1/base_objective_screen/modelscope
HFROOT=/NHNHOME/26msit001_A/BASE/AIPR/sjkim/revision_qwen3_8b/full_iter1/base_objective_screen/hf_ipv4
source "$VENV/bin/activate"
export HF_HUB_DISABLE_XET=1

wildguard() {
  ms download allenai/wildguard --repo-type model --revision master --local-dir "$MSROOT/wildguard" --max-workers 16 \
    --exclude 'original/*' '*.pth' '*.gguf' '*.bin' 'consolidated.safetensors' >"$OUT/wildguard.modelscope_rescue.log" 2>&1
  test -f "$MSROOT/wildguard/config.json"
  printf '%s\n' "$MSROOT/wildguard" >"$OUT/wildguard.path"
  printf 'wildguard\tallenai/wildguard\tmodelscope_master\tcomplete\n' >"$OUT/wildguard.status"
  date -Is >"$OUT/WILDGUARD_COMPLETE"
}

zephyr() {
  python "$PROJECT/analysis/qwen3_8b_base_objective_screen_20260716/download_hf_ipv4.py" \
    --repo HuggingFaceH4/zephyr-7b-beta --local-dir "$HFROOT/zephyr" --metadata "$OUT/metadata/zephyr.json" \
    >"$OUT/zephyr.hf_ipv4.log" 2>&1
  revision=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["resolved_revision"])' "$OUT/metadata/zephyr.json")
  printf '%s\n' "$HFROOT/zephyr" >"$OUT/zephyr.path"
  printf 'zephyr\tHuggingFaceH4/zephyr-7b-beta\t%s\tcomplete\n' "$revision" >"$OUT/zephyr.status"
  date -Is >"$OUT/ZEPHYR_COMPLETE"
}

wildguard & p0=$!
zephyr & p1=$!
wait "$p0" "$p1"
date -Is >"$OUT/RESCUE_COMPLETE"
