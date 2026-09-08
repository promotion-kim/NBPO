# Which capability evaluators can actually run, and what blocks the rest

Audited 2026-09-08 against the pod `nbpo-judge` (`p-aipr`, 3x H200) and the local
host. Every entry names the harness, the exact asset it needs, and where it
stands. The point is that a blocked benchmark is recorded as blocked rather than
quietly replaced by a local approximation.

`tab:general-capability` and the HarmBench column of `tab:saferlhf-main` are the
consumers.

| benchmark | harness | state | blocker |
|---|---|---|---|
| HarmBench ASR | `analysis/nbpo_v3_campaign_20260904/hb_new.sh`, official `cais/HarmBench-Llama-2-13b-cls` | **runnable** | none — classifier present on the PVC (25 GB), vLLM 0.10.1.1 on the pod |
| Beaver reward / cost | in-domain diagnostic, PKU reward+cost models | **runnable once the models are staged** | model weights not yet on this PVC; download is unblocked (public) |
| IFEval strict | EvalScope (`evalscope/run_rule_based_task.py`, `scripts/run_stage2_ifeval_suite.sh`) | **local only** | `evalscope` is installed in `/home/sjkim/anaconda3/envs/evalscope` on the host, not on the pod; the host's A100s are frequently held by another user |
| TruthfulQA MC2 | `lm_eval` (as in `scripts/revision/flagship/run_seed42_academic_suite.py`, 0-shot) | **blocked** | `/work/pylibs_lmeval` imports against transformers 5.16.1 and dies on `AutoModelForVision2Seq`, removed in transformers v5. Needs either a pinned transformers 4.x venv or a newer `lm_eval`. Not installed locally either. |
| Arena-Hard reference win | `evalscope/run_arena_hard_task.py` | **blocked** | official protocol judges with GPT-4; no OpenAI key available in this environment. A local judge would not be the official metric and must be labelled "reference win" if used at all |
| AlpacaEval-2 LC | none in this repo | **blocked** | harness not implemented here, and the official length-controlled metric needs a GPT-4 judge |

## Rules that apply to whatever does run

The prompted-Qwen judge is permanently retired: it may not appear in any final
evaluation, including as a stand-in judge for Arena-Hard or AlpacaEval. A local
pairwise score is reported under the name "reference win", never under the
official benchmark's name.

Beaver reward/cost is an **in-domain diagnostic**: it shares a distribution with
the training supervision and is not evidence of general transfer. It is reported
as such and never as an independent evaluation.

## Immediate consequence for the paper

Of the five columns in `tab:general-capability`, one (HarmBench ASR) is runnable
today, one (IFEval) needs a free local GPU, and three are blocked on either a
dependency conflict or an API key. The table therefore cannot be completed from
this environment alone, and the blocked cells stay em-dashed with this file as
the reason, rather than being filled with a local substitute metric.
