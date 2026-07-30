#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/MNPO_rev_20260710}
VENV=${VENV:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/venv_clean}
CACHE=${CACHE:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/revision_qwen3_8b/full_iter1/flagship_20260712/cache/huggingface}
OUT=${OUT:-$PROJECT/results/p1_8b_hh_selection_20260716/downloads}
mkdir -p "$OUT"
source "$VENV/bin/activate"
if [[ -f "$PROJECT/.secrets/hf.env" ]]; then
  set -a
  source "$PROJECT/.secrets/hf.env"
  set +a
fi

download_one() {
  local name=$1 repo=$2 revision=$3
  if hf download "$repo" --revision "$revision" --cache-dir "$CACHE" >"$OUT/$name.path" 2>"$OUT/$name.log"; then
    printf '%s\t%s\t%s\tcomplete\n' "$name" "$repo" "$revision" >"$OUT/$name.status"
  else
    local code=$?
    printf '%s\t%s\t%s\tfailed:%s\n' "$name" "$repo" "$revision" "$code" >"$OUT/$name.status"
    return "$code"
  fi
}
export -f download_one
export CACHE OUT

cat >"$OUT/download_grid.tsv" <<'EOF'
skywork_qwen3	Skywork/Skywork-Reward-V2-Qwen3-8B	6f19fdefb933293d4898bdb59a96f7223d998659
beaver_v1	PKU-Alignment/beaver-7b-v1.0-cost	c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28
beaver_v2	PKU-Alignment/beaver-7b-v2.0-cost	26bf7161f09fee958ae64c8b4bb70fb420f7ba39
llama_guard3	meta-llama/Llama-Guard-3-8B	7327bd9f6efbbe6101dc6cc4736302b3cbb6e425
shieldgemma	google/shieldgemma-9b	b8b636016df4540721a098c7aab91c97ec6ee508
qwen3guard8	Qwen/Qwen3Guard-Gen-8B	4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb
weak_small	Qwen/Qwen2.5-1.5B-Instruct	989aa7980e4cf806f80c7fef2b1adb7bc71aa306
EOF

cut -f1-3 "$OUT/download_grid.tsv" | xargs -P4 -n3 bash -c 'download_one "$0" "$1" "$2"'
date -Is >"$OUT/COMPLETE"
