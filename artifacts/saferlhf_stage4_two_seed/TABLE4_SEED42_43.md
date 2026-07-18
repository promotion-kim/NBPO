# Stage-4 SafeRLHF, training seeds 42 and 43

Values are the mean ± sample standard deviation across the two training seeds (n=2). Each seed was evaluated on the same 1,000 prompts with decode seed 42 and normalized independently over the same method pool.

| Method | Helpful. | Harmless. | Avg | Worst |
|---|---:|---:|---:|---:|
| RONPO (OS) | **0.6624 ± 0.0111** | 0.5856 ± 0.0016 | **0.6240 ± 0.0063** | **0.4412 ± 0.0085** |
| IPO | <u>0.5420 ± 0.0067</u> | <u>0.6539 ± 0.0003</u> | 0.5979 ± 0.0035 | <u>0.4046 ± 0.0027</u> |
| INPO-avg | 0.5023 ± 0.0390 | 0.6448 ± 0.0104 | 0.5736 ± 0.0143 | 0.3806 ± 0.0202 |
| DPO | 0.4978 ± 0.0140 | 0.6493 ± 0.0044 | 0.5736 ± 0.0092 | 0.3750 ± 0.0135 |
| SimPO | 0.4606 ± 0.0034 | **0.7410 ± 0.0116** | <u>0.6008 ± 0.0041</u> | 0.3730 ± 0.0002 |
| HT-MNPO (harml.) | 0.5242 ± 0.0123 | 0.4359 ± 0.0006 | 0.4801 ± 0.0064 | 0.2988 ± 0.0063 |
| HT-MNPO (help.) | 0.5329 ± 0.0171 | 0.4244 ± 0.0044 | 0.4786 ± 0.0064 | 0.2934 ± 0.0080 |
| SPPO-avg | 0.4856 ± 0.0026 | 0.4749 ± 0.0125 | 0.4803 ± 0.0050 | 0.2916 ± 0.0088 |
| Base | 0.2045 ± 0.0022 | 0.4456 ± 0.0012 | 0.3250 ± 0.0017 | 0.1315 ± 0.0009 |

Sample SD with two seeds is descriptive only; it is not a confidence interval or a stable variance estimate.
RONPO top-mass is retained in the full audit table as an estimator ablation and omitted from the main Table-4-style method table.
