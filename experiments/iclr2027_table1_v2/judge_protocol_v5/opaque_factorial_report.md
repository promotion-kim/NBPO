# Opaque-identifier factorial

balanced design verified from file: **True**

## scheme `ab`

unresolved invalid 0.0004  (first pass 0.0004)

| objective | position effect [95% CI] | identifier effect [95% CI] | tmpl range | conf swap | contra | order rho | determ | identical |
|---|---|---|---|---|---|---|---|---|
| helpfulness | +0.012 [-0.013, +0.037] | +0.237 [+0.183, +0.287] | 0.030 | 0.000 | 1.000 | -0.232 | 0.480 | 1.000 |
| honesty | -0.020 [-0.041, +0.000] | +0.017 [-0.010, +0.046] | 0.005 | 0.200 | 0.800 | 0.046 | 0.680 | 1.000 |
| instruction_following | +0.026 [+0.002, +0.051] | +0.147 [+0.086, +0.207] | 0.018 | 0.500 | 0.500 | -0.000 | 0.980 | 1.000 |
| truthfulness | -0.016 [-0.046, +0.013] | -0.019 [-0.072, +0.030] | 0.020 | 0.600 | 0.400 | 0.004 | 0.640 | 1.000 |

### gates

- **PASS**  `unresolved_invalid_rate` < 0.002  -> helpfulness=0.000, honesty=0.000, instruction_following=0.000, truthfulness=0.001
- **FAIL**  `deterministic_degradation_accuracy` >= 0.9  -> helpfulness=0.480, honesty=0.680, instruction_following=0.980, truthfulness=0.640
- **PASS**  `identical_confident_tie_accuracy` >= 0.9  -> helpfulness=1.000, honesty=1.000, instruction_following=1.000, truthfulness=1.000
- **FAIL**  `confident_semantic_swap_consistency` >= 0.85  -> helpfulness=0.000, honesty=0.200, instruction_following=0.500, truthfulness=0.600
- **FAIL**  `order_split_half_spearman` >= 0.75  -> helpfulness=-0.232, honesty=0.046, instruction_following=-0.000, truthfulness=0.004
- **NOT EVALUATED**  `finite_pool_target_pearson` >= 0.9
- **NOT EVALUATED**  `finite_pool_target_sign_agreement` >= 0.85

**Decision: B: permanently retire prompted-Qwen judging from the main experiment; no further prompt variants, no lowered thresholds**

## scheme `opaque`

unresolved invalid 0.0001  (first pass 0.0002)

| objective | position effect [95% CI] | identifier effect [95% CI] | tmpl range | conf swap | contra | order rho | determ | identical |
|---|---|---|---|---|---|---|---|---|
| helpfulness | +0.168 [+0.120, +0.219] | +0.068 [+0.042, +0.098] | 0.000 | 0.526 | 0.474 | 0.356 | 1.000 | 1.000 |
| honesty | -0.004 [-0.037, +0.029] | -0.016 [-0.036, +0.004] | 0.025 | 0.750 | 0.250 | 0.465 | 1.000 | 1.000 |
| instruction_following | +0.158 [+0.103, +0.215] | -0.005 [-0.026, +0.017] | 0.025 | 0.448 | 0.552 | 0.008 | 1.000 | 1.000 |
| truthfulness | +0.017 [-0.030, +0.064] | -0.017 [-0.043, +0.009] | 0.035 | 0.600 | 0.400 | 0.243 | 0.940 | 1.000 |

### gates

- **PASS**  `unresolved_invalid_rate` < 0.002  -> helpfulness=0.000, honesty=0.000, instruction_following=0.000, truthfulness=0.000
- **PASS**  `deterministic_degradation_accuracy` >= 0.9  -> helpfulness=1.000, honesty=1.000, instruction_following=1.000, truthfulness=0.940
- **PASS**  `identical_confident_tie_accuracy` >= 0.9  -> helpfulness=1.000, honesty=1.000, instruction_following=1.000, truthfulness=1.000
- **FAIL**  `confident_semantic_swap_consistency` >= 0.85  -> helpfulness=0.526, honesty=0.750, instruction_following=0.448, truthfulness=0.600
- **FAIL**  `order_split_half_spearman` >= 0.75  -> helpfulness=0.356, honesty=0.465, instruction_following=0.008, truthfulness=0.243
- **NOT EVALUATED**  `finite_pool_target_pearson` >= 0.9
- **NOT EVALUATED**  `finite_pool_target_sign_agreement` >= 0.85

**Decision: B: permanently retire prompted-Qwen judging from the main experiment; no further prompt variants, no lowered thresholds**
