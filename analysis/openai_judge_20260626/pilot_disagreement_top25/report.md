# GPT-5.5 Pairwise Judge Evaluation

Artifact directory: `analysis/openai_judge_20260626/pilot_disagreement_top25`

## Model Scoreboard

| Rank | Model | Mean pairwise win rate | Matchups |
| --- | --- | ---: | ---: |
| 1 | Base | 0.6585 | 5 |
| 2 | RONPO S2 checkpoint-2457 | 0.5556 | 5 |
| 3 | RONPO S2 checkpoint-1400 | 0.4971 | 5 |
| 4 | HT-MNPO Skywork S2 | 0.4747 | 5 |
| 5 | HT-MNPO ArmoRM S2 | 0.4345 | 5 |
| 6 | HT-MNPO Athene S2 | 0.3797 | 5 |

## Pairwise Results

| Left | Right | n | Left WR | Right WR | Tie | Confidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base | HT-MNPO ArmoRM S2 | 84 | 0.6369 | 0.3631 | 0.0595 | 0.8246 |
| Base | HT-MNPO Athene S2 | 90 | 0.7667 | 0.2333 | 0.0667 | 0.8154 |
| Base | HT-MNPO Skywork S2 | 95 | 0.6684 | 0.3316 | 0.1158 | 0.8205 |
| Base | RONPO S2 checkpoint-1400 | 93 | 0.6344 | 0.3656 | 0.0430 | 0.8132 |
| Base | RONPO S2 checkpoint-2457 | 87 | 0.5862 | 0.4138 | 0.0690 | 0.8090 |
| HT-MNPO ArmoRM S2 | RONPO S2 checkpoint-1400 | 85 | 0.4235 | 0.5765 | 0.0941 | 0.8153 |
| HT-MNPO ArmoRM S2 | RONPO S2 checkpoint-2457 | 88 | 0.3977 | 0.6023 | 0.0909 | 0.8174 |
| HT-MNPO Athene S2 | HT-MNPO ArmoRM S2 | 91 | 0.4725 | 0.5275 | 0.2198 | 0.8303 |
| HT-MNPO Athene S2 | RONPO S2 checkpoint-1400 | 84 | 0.4167 | 0.5833 | 0.0714 | 0.8208 |
| HT-MNPO Athene S2 | RONPO S2 checkpoint-2457 | 85 | 0.3647 | 0.6353 | 0.1176 | 0.8271 |
| HT-MNPO Skywork S2 | HT-MNPO ArmoRM S2 | 89 | 0.5393 | 0.4607 | 0.1348 | 0.8020 |
| HT-MNPO Skywork S2 | HT-MNPO Athene S2 | 90 | 0.5889 | 0.4111 | 0.2000 | 0.8269 |
| HT-MNPO Skywork S2 | RONPO S2 checkpoint-1400 | 94 | 0.4787 | 0.5213 | 0.1277 | 0.8246 |
| HT-MNPO Skywork S2 | RONPO S2 checkpoint-2457 | 92 | 0.4348 | 0.5652 | 0.2174 | 0.8205 |
| RONPO S2 checkpoint-1400 | RONPO S2 checkpoint-2457 | 98 | 0.4388 | 0.5612 | 0.4898 | 0.8593 |

Parse/API failures: `1056`. See `analysis_failures.json`.
