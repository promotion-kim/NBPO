# GPT-5.5 Judge Evaluation for RONPO Stage 2

Artifact directory: `analysis/openai_judge_20260626/full_647_final_paper_summary`

## Protocol

- Judge: `gpt-5.5-2026-04-23` via OpenAI Batch API.
- Compared models: Base, HT-MNPO Skywork/Athene/ArmoRM S2, RONPO S2 checkpoint-1400, RONPO S2 checkpoint-2457.
- Prompt set: 647 held-out UltraFeedback prompts from the existing stage-2 generation artifact.
- Comparison: all pairwise model pairs per prompt; response order deterministically randomized per prompt-pair.
- Win rate: ties count as 0.5; confidence intervals are prompt-level paired bootstrap intervals.

## Coverage

- Expected full judgments: `9705`.
- Parsed full judgments: `9705`.
- Missing/failed full judgments: `0`.
- Parsed high-disagreement judgments: `2430`.

## Full Held-Out Set

| Rank | Model | Mean pairwise win rate | 95% CI | Matchups |
| --- | --- | ---: | ---: | ---: |
| 1 | RONPO S2 checkpoint-2457 | 0.5927 | [0.5709, 0.6136] | 5 |
| 2 | Base | 0.5581 | [0.5329, 0.5841] | 5 |
| 3 | RONPO S2 checkpoint-1400 | 0.5532 | [0.5306, 0.5750] | 5 |
| 4 | HT-MNPO Skywork S2 | 0.5054 | [0.4841, 0.5272] | 5 |
| 5 | HT-MNPO ArmoRM S2 | 0.4263 | [0.4025, 0.4493] | 5 |
| 6 | HT-MNPO Athene S2 | 0.3643 | [0.3425, 0.3855] | 5 |

### Key Pairwise Results

| Pair | n | Left WR | 95% CI | Tie |
| --- | ---: | ---: | ---: | ---: |
| Base vs RONPO S2 checkpoint-1400 | 647 | 0.5108 | [0.4753, 0.5456] | 0.1314 |
| Base vs RONPO S2 checkpoint-2457 | 647 | 0.4822 | [0.4467, 0.5186] | 0.1360 |
| HT-MNPO ArmoRM S2 vs RONPO S2 checkpoint-2457 | 647 | 0.3617 | [0.3269, 0.3964] | 0.1051 |
| HT-MNPO Athene S2 vs RONPO S2 checkpoint-2457 | 647 | 0.3145 | [0.2805, 0.3486] | 0.1376 |
| HT-MNPO Skywork S2 vs RONPO S2 checkpoint-2457 | 647 | 0.4134 | [0.3802, 0.4467] | 0.1808 |
| RONPO S2 checkpoint-1400 vs RONPO S2 checkpoint-2457 | 647 | 0.4645 | [0.4320, 0.4961] | 0.3385 |

## High-Disagreement Stress Subset

| Rank | Model | Mean pairwise win rate | 95% CI | Matchups |
| --- | --- | ---: | ---: | ---: |
| 1 | Base | 0.6302 | [0.5802, 0.6772] | 5 |
| 2 | RONPO S2 checkpoint-2457 | 0.5241 | [0.4790, 0.5679] | 5 |
| 3 | RONPO S2 checkpoint-1400 | 0.4852 | [0.4414, 0.5272] | 5 |
| 4 | HT-MNPO ArmoRM S2 | 0.4809 | [0.4352, 0.5290] | 5 |
| 5 | HT-MNPO Skywork S2 | 0.4796 | [0.4377, 0.5185] | 5 |
| 6 | HT-MNPO Athene S2 | 0.4000 | [0.3574, 0.4389] | 5 |

## Conservative Interpretation

The full held-out GPT-5.5 judge evaluation is reward-model-independent evidence for stage-2 preference quality. It should be reported separately from local reward-model tables because the evaluator and metric differ. If coverage is complete or near-complete after retries, the result is appropriate for a main or appendix paper table.
