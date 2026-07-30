# GPT-5.5 Judge Evaluation for RONPO Stage 2

Artifact directory: `analysis/openai_judge_20260626/full_647_preliminary_paper_summary`

## Protocol

- Judge: `gpt-5.5-2026-04-23` via OpenAI Batch API.
- Compared models: Base, HT-MNPO Skywork/Athene/ArmoRM S2, RONPO S2 checkpoint-1400, RONPO S2 checkpoint-2457.
- Prompt set: 647 held-out UltraFeedback prompts from the existing stage-2 generation artifact.
- Comparison: all pairwise model pairs per prompt; response order deterministically randomized per prompt-pair.
- Win rate: ties count as 0.5; confidence intervals are prompt-level paired bootstrap intervals.

## Coverage

- Expected full judgments: `9705`.
- Parsed full judgments: `5829`.
- Missing/failed full judgments: `3876`.
- Parsed high-disagreement judgments: `1340`.

## Full Held-Out Set

| Rank | Model | Mean pairwise win rate | 95% CI | Matchups |
| --- | --- | ---: | ---: | ---: |
| 1 | RONPO S2 checkpoint-2457 | 0.6243 | [0.5929, 0.6535] | 5 |
| 2 | RONPO S2 checkpoint-1400 | 0.5920 | [0.5590, 0.6195] | 5 |
| 3 | Base | 0.5639 | [0.5292, 0.6000] | 5 |
| 4 | HT-MNPO Skywork S2 | 0.5033 | [0.4766, 0.5314] | 5 |
| 5 | HT-MNPO ArmoRM S2 | 0.3995 | [0.3698, 0.4303] | 5 |
| 6 | HT-MNPO Athene S2 | 0.3171 | [0.2886, 0.3500] | 5 |

### Key Pairwise Results

| Pair | n | Left WR | 95% CI | Tie |
| --- | ---: | ---: | ---: | ---: |
| Base vs RONPO S2 checkpoint-1400 | 391 | 0.5013 | [0.4551, 0.5462] | 0.0460 |
| Base vs RONPO S2 checkpoint-2457 | 389 | 0.4473 | [0.3898, 0.4947] | 0.0463 |
| HT-MNPO ArmoRM S2 vs RONPO S2 checkpoint-2457 | 391 | 0.3197 | [0.2741, 0.3675] | 0.0563 |
| HT-MNPO Athene S2 vs RONPO S2 checkpoint-2457 | 383 | 0.2546 | [0.2133, 0.2982] | 0.0809 |
| HT-MNPO Skywork S2 vs RONPO S2 checkpoint-2457 | 371 | 0.3827 | [0.3394, 0.4255] | 0.1240 |
| RONPO S2 checkpoint-1400 vs RONPO S2 checkpoint-2457 | 368 | 0.4742 | [0.4361, 0.5099] | 0.3723 |

## High-Disagreement Stress Subset

| Rank | Model | Mean pairwise win rate | 95% CI | Matchups |
| --- | --- | ---: | ---: | ---: |
| 1 | Base | 0.6699 | [0.6023, 0.7275] | 5 |
| 2 | RONPO S2 checkpoint-2457 | 0.5365 | [0.4744, 0.5995] | 5 |
| 3 | RONPO S2 checkpoint-1400 | 0.5066 | [0.4402, 0.5736] | 5 |
| 4 | HT-MNPO Skywork S2 | 0.4751 | [0.4100, 0.5436] | 5 |
| 5 | HT-MNPO ArmoRM S2 | 0.4372 | [0.3842, 0.5061] | 5 |
| 6 | HT-MNPO Athene S2 | 0.3747 | [0.3199, 0.4290] | 5 |

## Conservative Interpretation

The full held-out GPT-5.5 judge evaluation is reward-model-independent evidence for stage-2 preference quality. It should be reported separately from local reward-model tables because the evaluator and metric differ. If coverage is complete or near-complete after retries, the result is appropriate for a main or appendix paper table.
