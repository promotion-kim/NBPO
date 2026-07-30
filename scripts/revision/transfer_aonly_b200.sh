#!/usr/bin/env bash
# Poll ronpo2 for the finished aonly-b200 arm, then pull the final checkpoint
# to the local outputs dir so the factorized eval can score it.
set -uo pipefail
KEY=/home/sjkim/MNPO/nhn/aiprlab-ronpo2_key
SSHOPTS='-F /dev/null -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o LogLevel=ERROR -o ProxyCommand="ssh -F /dev/null -S /tmp/mnpo_havok_cm -o LogLevel=ERROR -W %h:%p sjkim@59.29.246.23 -p 3000"'
REMOTE=/NHNHOME/AIPR/sjkim/ronpo_arms_20260720/os-ronpo-aonly-k05-b200
LOCAL=/ext_hdd/sjkim/mnpo/outputs_ronpo_h200/os_ronpo_aonly_k05_b200

while ! eval ssh $SSHOPTS -i $KEY -p 31712 aipr_lab@59.150.33.1 "test -f $REMOTE/all_results.json" 2>/dev/null; do
  sleep 600
done
echo "[transfer] aonly-b200 finished at $(date -Is); pulling checkpoint"
mkdir -p "$LOCAL"
eval rsync -az --info=stats1 -e \"ssh $SSHOPTS -i $KEY -p 31712\" \
  --exclude 'checkpoint-*' --exclude 'global_step*' --exclude 'wandb' \
  aipr_lab@59.150.33.1:$REMOTE/ "$LOCAL/"
echo "[transfer] done at $(date -Is): $(ls $LOCAL | tr '\n' ' ')"
