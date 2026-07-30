#!/usr/bin/env bash
set -euo pipefail
PROJECT=${PROJECT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/venv_clean}
CACHE=${CACHE:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/revision_qwen3_8b/full_iter1/base_objective_screen/cache}
OUT=$PROJECT/results/p1_8b_base_objective_screen_20260716/downloads
mkdir -p "$OUT" "$CACHE"
source "$VENV/bin/activate"
if [[ -f "$PROJECT/.secrets/hf.env" ]]; then
  set -a; source "$PROJECT/.secrets/hf.env"; set +a
fi

download_one() {
  local name=$1 repo=$2
  local attempt
  for attempt in 1 2 3 4 5; do
    printf '%s\t%s\t%s\tattempt=%s\n' "$(date -Is)" "$name" "$repo" "$attempt" >>"$OUT/retries.log"
    if hf download "$repo" --cache-dir "$CACHE" >"$OUT/$name.path.tmp" 2>"$OUT/$name.attempt${attempt}.log"; then
      mv "$OUT/$name.path.tmp" "$OUT/$name.path"
      local snapshot
      snapshot=$(tail -1 "$OUT/$name.path" | sed -n 's/^  path: //p; t; p')
      if [[ -z "$snapshot" || ! -f "$snapshot/config.json" ]]; then
        printf '%s invalid_snapshot_path\n' "$name" >>"$OUT/retries.log"
        sleep 30
        continue
      fi
      printf '%s\t%s\t%s\tcomplete\n' "$name" "$repo" "$(basename "$snapshot")" >"$OUT/$name.status"
      return 0
    fi
    sleep 30
  done
  printf '%s\t%s\tunknown\tfailed_after_5_attempts\n' "$name" "$repo" >"$OUT/$name.status"
  return 1
}
export -f download_one
export CACHE OUT

cat >"$OUT/grid.tsv" <<'EOF'
llama31	meta-llama/Llama-3.1-8B-Instruct
qwen25	Qwen/Qwen2.5-7B-Instruct
mistral7	mistralai/Mistral-7B-Instruct-v0.3
wildguard	allenai/wildguard
zephyr	HuggingFaceH4/zephyr-7b-beta
EOF

cut -f1-2 "$OUT/grid.tsv" | xargs -P4 -n2 bash -c 'download_one "$0" "$1"'
date -Is >"$OUT/COMPLETE"
