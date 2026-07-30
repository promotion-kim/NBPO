#!/usr/bin/env bash
# Eval HH stage-2 arms (unif,nbs) with the LLM judge: decode on test (seed42,43),
# judge vs the SFT-base test decodes reused from stage-1, surplus=u_k. Runs on ronpo GPU1-3.
set -euo pipefail
P=/NHNHOME/AIPR/sjkim/MNPO_rev_20260710
V=/NHNHOME/AIPR/sjkim/venv_clean
R=/NHNHOME/AIPR/sjkim/nbpo_hh_sft_20260726
J=/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1/flagship_20260712/cache/huggingface/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
DEC=$P/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py
MAN=$R/hh_test_prompts.jsonl
OBJ=helpfulness,harmlessness,humor
E=$R/bpo/s2
BREF=$R/bpo/eval/base/gens   # reuse stage-1 base test decodes seed44/45
export PYTHONPATH=$P HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1
dec(){ local model=$1 name=$2 s=$3 g=$4; local out=$E/eval/$name/gens/seed$s/output_$s.json
  mkdir -p "$(dirname "$out")"; [ -s "$out" ] && return
  CUDA_VISIBLE_DEVICES=$g $V/bin/python $DEC --manifest $MAN --model "$model" --policy-name ${name}s2_$s \
    --probe none --output "$out" --seed $s --temperature 0.7 --top-p 0.9 --max-new-tokens 512 \
    --max-model-len 2048 --gpu-memory-utilization 0.85 > $E/eval/$name/dec_$s.log 2>&1; }
echo "[s2-eval] DECODE"
dec $E/train/unif unif 42 1 & dec $E/train/unif unif 43 2 & dec $E/train/nbs nbs 42 3 & wait
dec $E/train/nbs nbs 43 1 & wait
echo "[s2-eval] DECODE_DONE"
for arm in unif nbs; do
  [ -s $E/eval/$arm/surplus.json ] && continue
  VD=$E/eval/$arm/verdicts; mkdir -p $VD
  for spec in "1 0" "2 1" "3 2"; do set -- $spec; g=$1; idx=$2
    tmux new-session -d -s je${arm}$idx "CUDA_VISIBLE_DEVICES=$g $V/bin/python $P/scripts/bpo/judge_bpo.py \
      --policy-files 42=$E/eval/$arm/gens/seed42/output_42.json 43=$E/eval/$arm/gens/seed43/output_43.json \
      --reference-files 44=$BREF/seed44/output_44.json 45=$BREF/seed45/output_45.json \
      --judge-model-path $J --objectives $OBJ --output $VD/shard$idx.jsonl --num-shards 3 --shard-index $idx \
      > $VD/j$idx.log 2>&1"
  done
  sleep 8; while tmux ls 2>/dev/null | grep -q "^je$arm"; do sleep 20; done
  cat $VD/shard*.jsonl > $E/eval/$arm/verdicts.jsonl
  $V/bin/python $P/scripts/bpo/eval_bpo_surplus.py --verdicts $E/eval/$arm/verdicts.jsonl --label $arm --out $E/eval/$arm/surplus.json
done
echo "S2_EVAL_DONE" > $E/EVAL_DONE
