#!/usr/bin/env bash
# Eval the 4 BPO arms with the LLM judge: decode arms(seed42,43)+base(seed44,45) on HH
# test prompts, judge each arm vs base per objective (Qwen3-32B, swap-avg), surplus=u_k.
# Runs on one 4-GPU host. Usage: hh_bpo_eval.sh
set -euo pipefail
P=/NHNHOME/AIPR/sjkim/MNPO_rev_20260710
V=/NHNHOME/AIPR/sjkim/venv_clean
BASE=/NHNHOME/AIPR/sjkim/sft_bases/zephyr-7b-sft-full
R=/NHNHOME/AIPR/sjkim/nbpo_hh_sft_20260726
J=/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1/flagship_20260712/cache/huggingface/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
DEC=$P/analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py
MAN=$R/hh_test_prompts.jsonl
OBJ=helpfulness,harmlessness,humor
E=$R/bpo
export PYTHONPATH=$P HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_CUDNN_SDPA_ENABLED=0 MNPO_DISABLE_CUDNN_SDPA=1

dec(){ local model=$1 name=$2 s=$3 g=$4; local out=$E/eval/$name/gens/seed$s/output_$s.json
  mkdir -p "$(dirname "$out")"; [ -s "$out" ] && return
  CUDA_VISIBLE_DEVICES=$g $V/bin/python $DEC --manifest $MAN --model "$model" --policy-name ${name}_$s \
    --probe none --output "$out" --seed $s --temperature 0.7 --top-p 0.9 --max-new-tokens 512 \
    --max-model-len 2048 --gpu-memory-utilization 0.85 > $E/eval/$name/dec_$s.log 2>&1; }

echo "[bpo-eval] DECODE"
dec $BASE base 44 0 & dec $BASE base 45 1 & dec $E/train/unif unif 42 2 & dec $E/train/unif unif 43 3 & wait
dec $E/train/nbs nbs 42 0 & dec $E/train/nbs nbs 43 1 & dec $E/train/ks ks 42 2 & dec $E/train/ks ks 43 3 & wait
dec $E/train/maxmin maxmin 42 0 & dec $E/train/maxmin maxmin 43 1 & wait
echo "[bpo-eval] DECODE_DONE"

for arm in unif nbs ks maxmin; do
  [ -s $E/eval/$arm/surplus.json ] && continue
  VD=$E/eval/$arm/verdicts; mkdir -p $VD
  for g in 0 1 2 3; do
    tmux new-session -d -s je${arm}$g "CUDA_VISIBLE_DEVICES=$g $V/bin/python $P/scripts/bpo/judge_bpo.py \
      --policy-files 42=$E/eval/$arm/gens/seed42/output_42.json 43=$E/eval/$arm/gens/seed43/output_43.json \
      --reference-files 44=$E/eval/base/gens/seed44/output_44.json 45=$E/eval/base/gens/seed45/output_45.json \
      --judge-model-path $J --objectives $OBJ --output $VD/shard$g.jsonl --num-shards 4 --shard-index $g \
      > $VD/j$g.log 2>&1"
  done
  sleep 8; while tmux ls 2>/dev/null | grep -q "^je$arm"; do sleep 20; done
  cat $VD/shard*.jsonl > $E/eval/$arm/verdicts.jsonl
  $V/bin/python $P/scripts/bpo/eval_bpo_surplus.py --verdicts $E/eval/$arm/verdicts.jsonl \
    --label $arm --out $E/eval/$arm/surplus.json
done
echo "BPO_EVAL_DONE" > $E/EVAL_DONE
