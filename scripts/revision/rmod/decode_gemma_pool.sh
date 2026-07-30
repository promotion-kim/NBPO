#!/usr/bin/env bash
# Stage-1 response-pool generation for RONPO-on-gemma-2-2b-it (RMOD-matched
# setting). Decodes UltraFeedback train+test prompts with gemma-2-2b-it across
# 4 B200 GPUs (one seed group per GPU), for the 5-ArmoRM-head RONPO run.
#   bash decode_gemma_pool.sh          # run on the ronpo host (4 GPUs)
set -uo pipefail
SJ=/NHNHOME/AIPR/sjkim
PROJECT=$SJ/MNPO_rev_20260720
WORK=$SJ/ronpo_gemma_20260720
PY=$SJ/venv_clean/bin/python
export PYTHONPATH=$PROJECT
export HF_HOME=$SJ/baseline_repair_1p5b_20260714/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TOKENIZERS_PARALLELISM=false
mkdir -p $WORK/pool/logs
MODEL=google/gemma-2-2b-it

decode() { # gpu split seedlist datafile
  local gpu=$1 split=$2 seeds=$3 data=$4
  CUDA_VISIBLE_DEVICES=$gpu $PY -u -m on_policy_data_gen.decode \
    --data_dir "$data" --model "$MODEL" --seeds $seeds \
    --output_dir "$WORK/pool/$split/gpu$gpu" --num_gpu 1 \
    --temperature 0.8 --top_p 0.95 --max_tokens 1024 --batch_size 512 \
    --dtype bfloat16 --cache_dir "$HF_HUB_CACHE" \
    > "$WORK/pool/logs/decode_${split}_gpu${gpu}.log" 2>&1
}

TR=$PROJECT/data/gemma2_ufb_part2_train.jsonl
TE=$PROJECT/data/gemma2_ufb_part2_test.jsonl
# train pool: 5 seeds over GPUs 0-3 (gpu0 takes 2)
decode 0 train "13 21" "$TR" & p0=$!
decode 1 train "42"    "$TR" & p1=$!
decode 2 train "79"    "$TR" & p2=$!
decode 3 train "100"   "$TR" & p3=$!
wait $p0 $p1 $p2 $p3
# test pool (small): split same way
decode 0 test "13 21" "$TE" & q0=$!
decode 1 test "42"    "$TE" & q1=$!
decode 2 test "79"    "$TE" & q2=$!
decode 3 test "100"   "$TE" & q3=$!
wait $q0 $q1 $q2 $q3
echo "[pool] gemma-2-2b-it decode done at $(date -Is)"
