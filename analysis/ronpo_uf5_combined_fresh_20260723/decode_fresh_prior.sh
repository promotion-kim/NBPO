#!/usr/bin/env bash
set -euo pipefail
SJ=/NHNHOME/AIPR/sjkim
P=$SJ/MNPO_rev_20260720
ROOT=$SJ/ronpo_uf5_combined_fresh_20260723
PY=$SJ/venv_clean/bin/python
CACHE=$SJ/baseline_repair_1p5b_20260714/cache/huggingface/hub
DATA=$P/data/gemma2_ufb_part3_test.jsonl
REPO=promotion/ronpo-gemma2-2b-uf5-anneal-s42
export PYTHONPATH=$P HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface
export HF_HUB_CACHE=$CACHE HUGGINGFACE_HUB_CACHE=$CACHE TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/fresh/generations" "$ROOT/fresh/downloads" "$ROOT/fresh/logs" "$ROOT/audit"

decode() {
  local tag=$1 model=$2
  mkdir -p "$ROOT/fresh/generations/$tag"
  if [[ ! -s "$ROOT/fresh/generations/$tag/output_42.json" ]]; then
    CUDA_VISIBLE_DEVICES=1 "$PY" -u -m on_policy_data_gen.decode \
      --data_dir "$DATA" --model "$model" --seeds 42 \
      --output_dir "$ROOT/fresh/generations/$tag" --num_gpu 1 --temperature 0.8 \
      --top_p 0.95 --max_tokens 128 --batch_size 256 --dtype bfloat16 --cache_dir "$CACHE" \
      > "$ROOT/fresh/logs/decode_${tag}.log" 2>&1
  fi
  "$PY" - "$ROOT/fresh/generations/$tag/output_42.json" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1])); assert len(rows)==647, len(rows)
PY
}

decode base google/gemma-2-2b-it

tags=(ss_s1 ss_s2 ss_s3 ss_s4)
subs=(stronger_signal/stage1 stronger_signal/stage2 stronger_signal/stage3 stronger_signal/stage4)
revs=(
  190744cc90bec6b8d8dd188656b2bbbf5787779d
  51cc838723d3f983dd946a8493ba5542e9ddca94
  c44d2b52cfeb76dcdb071b30a2d31342848d17c1
  d06cff7476d54571ed76e6ce606360176ff31c24
)
for i in 0 1 2 3; do
  if [[ -s "$ROOT/fresh/generations/${tags[$i]}/output_42.json" ]] && \
     "$PY" -c 'import json,sys; assert len(json.load(open(sys.argv[1]))) == 647' "$ROOT/fresh/generations/${tags[$i]}/output_42.json"; then
    continue
  fi
  tmp="$ROOT/fresh/downloads/${tags[$i]}"
  model=
  for attempt in 1 2 3 4 5; do
    if model=$($PY "$P/analysis/ronpo_uf5_combined_fresh_20260723/download_subfolder.py" \
      --repo "$REPO" --revision "${revs[$i]}" --subfolder "${subs[$i]}" --output "$tmp"); then
      break
    fi
    sleep $((attempt * 10))
  done
  [[ -n $model ]] || { echo "failed to download ${tags[$i]} after five transport retries" >&2; exit 1; }
  decode "${tags[$i]}" "$model"
  "$PY" - "${tags[$i]}" "${subs[$i]}" "${revs[$i]}" \
    "$ROOT/fresh/generations/${tags[$i]}/output_42.json" "$ROOT/audit/fresh_prior.jsonl" <<'PY'
import hashlib,json,sys
tag,sub,rev,path,out=sys.argv[1:]
h=hashlib.sha256(open(path,'rb').read()).hexdigest()
with open(out,'a') as f: f.write(json.dumps({'tag':tag,'subfolder':sub,'revision':rev,'rows':647,'generation_sha256':h})+'\n')
PY
  rm -rf "$tmp"
done
date -Is > "$ROOT/fresh/PRIOR_DECODE_COMPLETE"
