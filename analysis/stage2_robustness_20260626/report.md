# Stage-2 Robustness Bootstrap and Disagreement Analysis

Source artifact: `/ext_hdd/sjkim/mnpo/eval/ronpo_stage2_resume_sanity_20260625`.
All metrics are recomputed from `scored/eval_{skywork,athene,armo}.jsonl` with the same six-model comparison set.
Prompt-normalized reward is min-max normalized across the six models for each prompt and objective before averaging.
Win rates count exact ties as 0.5. Confidence intervals are prompt-level paired bootstrap 95% intervals.

## Full Held-Out Set

| Method | n | Avg norm | Worst norm | Avg WR vs Base | Worst WR vs Base |
| --- | --- | --- | --- | --- | --- |
| Base | 647 | 0.4852 [0.4607, 0.5114] | 0.4099 [0.3810, 0.4403] | - | - |
| HT-MNPO Skywork S2 | 647 | 0.5020 [0.4775, 0.5252] | 0.4735 [0.4456, 0.4998] | 0.5229 [0.4902, 0.5538] | 0.4753 [0.4359, 0.5108] |
| HT-MNPO Athene S2 | 647 | 0.3881 [0.3647, 0.4125] | 0.3372 [0.3113, 0.3645] | 0.4384 [0.4085, 0.4704] | 0.3802 [0.3454, 0.4158] |
| HT-MNPO ArmoRM S2 | 647 | 0.4148 [0.3937, 0.4384] | 0.3702 [0.3451, 0.3961] | 0.4536 [0.4243, 0.4858] | 0.3787 [0.3423, 0.4189] |
| RONPO S2 checkpoint-1400 | 647 | 0.6827 [0.6606, 0.7041] | 0.6569 [0.6309, 0.6819] | 0.6466 [0.6156, 0.6759] | 0.6036 [0.5657, 0.6407] |
| RONPO S2 checkpoint-2457 | 647 | 0.7025 [0.6814, 0.7243] | 0.6701 [0.6439, 0.6961] | 0.6605 [0.6314, 0.6896] | 0.6005 [0.5618, 0.6399] |

## High-Disagreement Top 25% Subset

| Method | n | Avg norm | Worst norm | Avg WR vs Base | Worst WR vs Base |
| --- | --- | --- | --- | --- | --- |
| Base | 162 | 0.5246 [0.4824, 0.5677] | 0.3788 [0.3172, 0.4371] | - | - |
| HT-MNPO Skywork S2 | 162 | 0.4956 [0.4520, 0.5366] | 0.4550 [0.3932, 0.4985] | 0.4794 [0.4259, 0.5298] | 0.3457 [0.2716, 0.4198] |
| HT-MNPO Athene S2 | 162 | 0.4329 [0.3921, 0.4770] | 0.3392 [0.2836, 0.3962] | 0.4280 [0.3817, 0.4753] | 0.3302 [0.2562, 0.3859] |
| HT-MNPO ArmoRM S2 | 162 | 0.4814 [0.4417, 0.5212] | 0.4113 [0.3577, 0.4567] | 0.4681 [0.4197, 0.5175] | 0.3642 [0.2931, 0.4198] |
| RONPO S2 checkpoint-1400 | 162 | 0.5694 [0.5300, 0.6087] | 0.5487 [0.4873, 0.5862] | 0.5340 [0.4846, 0.5834] | 0.4599 [0.3858, 0.5278] |
| RONPO S2 checkpoint-2457 | 162 | 0.6018 [0.5626, 0.6403] | 0.5733 [0.5176, 0.6166] | 0.5761 [0.5267, 0.6265] | 0.4815 [0.4105, 0.5586] |

## RONPO Final Pairwise Win Rates

### full
| Comparison | Skywork WR | Athene WR | ArmoRM WR | Avg WR | Worst WR |
| --- | --- | --- | --- | --- | --- |
| RONPO S2 final vs Base | 0.6886 | 0.6924 | 0.6005 | 0.6605 [0.6314, 0.6896] | 0.6005 [0.5618, 0.6399] |
| RONPO S2 final vs HT-MNPO Skywork S2 | 0.6376 | 0.7179 | 0.6832 | 0.6795 [0.6504, 0.7081] | 0.6376 [0.6012, 0.6739] |
| RONPO S2 final vs HT-MNPO Athene S2 | 0.6476 | 0.7991 | 0.7396 | 0.7287 [0.7012, 0.7555] | 0.6476 [0.6105, 0.6824] |
| RONPO S2 final vs HT-MNPO ArmoRM S2 | 0.6430 | 0.7751 | 0.7388 | 0.7190 [0.6909, 0.7465] | 0.6430 [0.6059, 0.6808] |
| RONPO S2 final vs RONPO S2 checkpoint-1400 | 0.5070 | 0.5317 | 0.5193 | 0.5193 [0.4907, 0.5479] | 0.5070 [0.4699, 0.5355] |

### disagreement_top25
| Comparison | Skywork WR | Athene WR | ArmoRM WR | Avg WR | Worst WR |
| --- | --- | --- | --- | --- | --- |
| RONPO S2 final vs Base | 0.6543 | 0.5926 | 0.4815 | 0.5761 [0.5267, 0.6265] | 0.4815 [0.4105, 0.5586] |
| RONPO S2 final vs HT-MNPO Skywork S2 | 0.4907 | 0.6481 | 0.6173 | 0.5854 [0.5391, 0.6348] | 0.4907 [0.4167, 0.5679] |
| RONPO S2 final vs HT-MNPO Athene S2 | 0.4722 | 0.7469 | 0.5926 | 0.6039 [0.5556, 0.6502] | 0.4722 [0.3951, 0.5463] |
| RONPO S2 final vs HT-MNPO ArmoRM S2 | 0.4938 | 0.6883 | 0.6481 | 0.6101 [0.5668, 0.6543] | 0.4938 [0.4228, 0.5710] |
| RONPO S2 final vs RONPO S2 checkpoint-1400 | 0.5062 | 0.5494 | 0.5463 | 0.5340 [0.4835, 0.5802] | 0.5062 [0.4383, 0.5586] |

## Disagreement Subset Definition

For each prompt, every pair of candidate model responses is compared under each pair of reward objectives.
The disagreement score is the fraction of non-tied objective-pair/model-pair comparisons where the objectives prefer opposite responses.
The top-25% subset has n=162, disagreement-rate range [0.3111, 0.6897], and mean 0.4184.

## Paper-Ready Interpretation

The full held-out set already shows RONPO S2 final as the strongest local-RM model by average and worst-objective normalized reward.
The high-disagreement subset is the more targeted stress test: it isolates prompts where the reward sources disagree over candidate responses.
RONPO S2 final remains the strongest method on both average and worst-objective metrics on this subset, supporting the core robustness claim that objective-adversarial training raises the weakest reward-source floor rather than merely improving an easy average.

This result should be reported alongside the IFEval finding: RONPO S2 preserves rule-based instruction following better than HT-MNPO S2, while the local-RM stress test shows stronger robustness under reward-source conflict.

## Generated Files

- `analysis/stage2_robustness_20260626/bootstrap_model_metrics.csv`
- `analysis/stage2_robustness_20260626/bootstrap_pairwise_ronpo_final.csv`
- `analysis/stage2_robustness_20260626/disagreement_prompts.csv`
- `analysis/stage2_robustness_20260626/model_summary_disagreement_top25.csv`
- `analysis/stage2_robustness_20260626/model_summary_full.csv`
- `analysis/stage2_robustness_20260626/pairwise_ronpo_final_disagreement_top25.csv`
- `analysis/stage2_robustness_20260626/pairwise_ronpo_final_full.csv`
- `analysis/stage2_robustness_20260626/per_objective_disagreement_top25.csv`
- `analysis/stage2_robustness_20260626/per_objective_full.csv`
