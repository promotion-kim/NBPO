#!/usr/bin/env bash
cd /home/sjkim/MNPO; export HF_HOME=/ext_hdd/sjkim/hf_cache
PY=/home/sjkim/anaconda3/envs/mnpo_infer/bin/python
DEC=analysis/qwen3_8b_hh_selection_20260716/decode_policy_screen.py
MAN=/home/sjkim/MNPO/local_xj/uf_test.jsonl; O=/home/sjkim/MNPO/local_xj
TOP=/ext_hdd/sjkim/mnpo/outputs_ronpo_h200/os_ronpo_topmass_k05
OS=/ext_hdd/sjkim/mnpo/outputs_ronpo_h200/os_ronpo_os_k05
BASE=/ext_hdd/sjkim/huggingface/transformers/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
dec(){ CUDA_VISIBLE_DEVICES=$4 $PY $DEC --manifest $MAN --model $1 --policy-name $2_$3 --probe none --output $O/gen_$2_$3.json --seed $3 --temperature 0.7 --top-p 0.9 --max-new-tokens 400 --max-model-len 2048 --gpu-memory-utilization 0.55 > $O/dec_$2_$3.log 2>&1; }
dec $TOP topmass 42 0 & dec $OS os 42 1 & dec $BASE base 44 2 & wait
dec $TOP topmass 43 0 & dec $OS os 43 1 & dec $BASE base 45 2 & wait
echo DECODE_DONE > $O/DECODE_DONE
