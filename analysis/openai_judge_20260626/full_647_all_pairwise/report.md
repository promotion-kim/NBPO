# GPT-5.5 Pairwise Judge Evaluation

Artifact directory: `analysis/openai_judge_20260626/full_647_all_pairwise`

## Model Scoreboard

| Rank | Model | Mean pairwise win rate | Matchups |
| --- | --- | ---: | ---: |
| 1 | RONPO S2 checkpoint-2457 | 0.5927 | 5 |
| 2 | Base | 0.5581 | 5 |
| 3 | RONPO S2 checkpoint-1400 | 0.5532 | 5 |
| 4 | HT-MNPO Skywork S2 | 0.5054 | 5 |
| 5 | HT-MNPO ArmoRM S2 | 0.4263 | 5 |
| 6 | HT-MNPO Athene S2 | 0.3643 | 5 |

## Pairwise Results

| Left | Right | n | Left WR | Right WR | Tie | Confidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base | HT-MNPO ArmoRM S2 | 647 | 0.5927 | 0.4073 | 0.1036 | 0.7673 |
| Base | HT-MNPO Athene S2 | 647 | 0.6430 | 0.3570 | 0.1175 | 0.7645 |
| Base | HT-MNPO Skywork S2 | 647 | 0.5618 | 0.4382 | 0.1159 | 0.7604 |
| Base | RONPO S2 checkpoint-1400 | 647 | 0.5108 | 0.4892 | 0.1314 | 0.7597 |
| Base | RONPO S2 checkpoint-2457 | 647 | 0.4822 | 0.5178 | 0.1360 | 0.7647 |
| HT-MNPO ArmoRM S2 | RONPO S2 checkpoint-1400 | 647 | 0.3833 | 0.6167 | 0.1360 | 0.7549 |
| HT-MNPO ArmoRM S2 | RONPO S2 checkpoint-2457 | 647 | 0.3617 | 0.6383 | 0.1051 | 0.7560 |
| HT-MNPO Athene S2 | HT-MNPO ArmoRM S2 | 647 | 0.4436 | 0.5564 | 0.2040 | 0.7695 |
| HT-MNPO Athene S2 | RONPO S2 checkpoint-1400 | 647 | 0.3470 | 0.6530 | 0.1592 | 0.7548 |
| HT-MNPO Athene S2 | RONPO S2 checkpoint-2457 | 647 | 0.3145 | 0.6855 | 0.1376 | 0.7688 |
| HT-MNPO Skywork S2 | HT-MNPO ArmoRM S2 | 647 | 0.5773 | 0.4227 | 0.1808 | 0.7526 |
| HT-MNPO Skywork S2 | HT-MNPO Athene S2 | 647 | 0.6406 | 0.3594 | 0.1654 | 0.7586 |
| HT-MNPO Skywork S2 | RONPO S2 checkpoint-1400 | 647 | 0.4575 | 0.5425 | 0.1669 | 0.7473 |
| HT-MNPO Skywork S2 | RONPO S2 checkpoint-2457 | 647 | 0.4134 | 0.5866 | 0.1808 | 0.7473 |
| RONPO S2 checkpoint-1400 | RONPO S2 checkpoint-2457 | 647 | 0.4645 | 0.5355 | 0.3385 | 0.7584 |
