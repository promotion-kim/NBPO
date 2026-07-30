#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../../.." && pwd)
BSSH=$HERE/nhn/bssh.sh
while ! "$BSSH" ronpo 'test -s /NHNHOME/AIPR/sjkim/MNPO_rev_20260710/results/rmod_figure3_kappa_20260721/eval_joint/COMPLETED'; do sleep 60; done
scp -F /dev/null -i "$HERE/nhn/aiprlab-ronpo_key" -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o LogLevel=ERROR \
  -o ProxyCommand='ssh -F /dev/null -S /tmp/mnpo_havok_cm -o LogLevel=ERROR -W %h:%p sjkim@59.29.246.23 -p 3000' -P 31603 \
  aipr_lab@59.150.33.1:/NHNHOME/AIPR/sjkim/MNPO_rev_20260710/results/rmod_figure3_kappa_20260721/eval_joint/saferlhf_kappa_stage_front.pdf \
  "$HERE/ronpo_aaai/figures/saferlhf_stage_front.pdf"
while ! "$BSSH" ronpo 'test -s /NHNHOME/AIPR/sjkim/ronpo_gemma_final_20260722/COMPLETED'; do sleep 60; done
scp -F /dev/null -i "$HERE/nhn/aiprlab-ronpo_key" -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o LogLevel=ERROR \
  -o ProxyCommand='ssh -F /dev/null -S /tmp/mnpo_havok_cm -o LogLevel=ERROR -W %h:%p sjkim@59.29.246.23 -p 3000' -P 31603 \
  aipr_lab@59.150.33.1:/NHNHOME/AIPR/sjkim/ronpo_gemma_final_20260722/uf5_radar.pdf \
  "$HERE/ronpo_aaai/figures/uf5_radar.pdf"
cd "$HERE/ronpo_aaai"
PATH="$HERE/.TinyTeX/bin/x86_64-linux:$PATH" pdflatex -interaction=nonstopmode -halt-on-error main_v3.tex > "$HERE/results/rmod_stage_extension_20260721/final_compile.log" 2>&1
date -Is > "$HERE/results/rmod_stage_extension_20260721/FIGURES_SYNCED"
