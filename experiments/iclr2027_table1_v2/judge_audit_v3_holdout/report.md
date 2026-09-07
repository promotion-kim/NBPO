# Section F — frozen-protocol holdout audit

Protocol: `P2_deliberative`  
Pairs: **15245**, judge calls **56548**, wall clock 1449.8 s

## Gates: **FAIL**

- FAIL: helpfulness: invalid rate 4.0000%
- FAIL: helpfulness: |position bias| 0.156
- FAIL: helpfulness: confident-decisive swap consistency 0.727
- FAIL: helpfulness: split-half Spearman 0.423
- FAIL: helpfulness: unresolved uncertainty 0.298
- FAIL: honesty: invalid rate 13.0909%
- FAIL: honesty: |position bias| 0.049
- FAIL: honesty: confident-decisive swap consistency 0.774
- FAIL: honesty: split-half Spearman 0.437
- FAIL: honesty: unresolved uncertainty 0.262
- FAIL: instruction_following: invalid rate 15.5227%
- FAIL: instruction_following: |position bias| 0.193
- FAIL: instruction_following: confident-decisive swap consistency 0.703
- FAIL: instruction_following: split-half Spearman 0.426
- FAIL: instruction_following: unresolved uncertainty 0.318
- FAIL: truthfulness: invalid rate 20.9091%
- FAIL: truthfulness: |position bias| 0.040
- FAIL: truthfulness: confident-decisive swap consistency 0.772
- FAIL: truthfulness: split-half Spearman 0.432
- FAIL: truthfulness: unresolved uncertainty 0.250

## Per objective

| objective | pairs | invalid | adjud. | unresolved | raw swap | **confident swap** | contradiction | position bias (95% CI) | length bias | mean \|D\| | conf. tie |
|---|---|---|---|---|---|---|---|---|---|---|---|
| helpfulness | 4224 | 4.000% | 0.343 | 0.298 | 0.727 | **0.727** | 0.273 | +0.1565 [0.1450, 0.1679] | 0.059 | 0.3164 | 0.119 |
| honesty | 3824 | 13.091% | 0.346 | 0.262 | 0.768 | **0.774** | 0.232 | +0.0494 [0.0386, 0.0608] | -0.064 | 0.2790 | 0.222 |
| instruction_following | 3717 | 15.523% | 0.382 | 0.318 | 0.702 | **0.703** | 0.298 | +0.1926 [0.1799, 0.2038] | -0.102 | 0.2805 | 0.173 |
| truthfulness | 3480 | 20.909% | 0.330 | 0.250 | 0.769 | **0.772** | 0.231 | +0.0397 [0.0286, 0.0507] | -0.058 | 0.2605 | 0.261 |

## Reliability

| objective | split-half (order) rho | split-half (template) rho | n template pairs |
|---|---|---|---|
| helpfulness | 0.585 | 0.423 | 1440 |
| honesty | 0.618 | 0.437 | 1292 |
| instruction_following | 0.543 | 0.426 | 1382 |
| truthfulness | 0.585 | 0.432 | 1054 |

## Cycles and transitivity

| objective | reliable triples | hard cycle rate | soft cycle mass | BT deviance |
|---|---|---|---|---|
| instruction_following | 474 | 0.0063 | 0.0571 | 0.0726 |
| truthfulness | 438 | 0.0000 | 0.0705 | 0.0758 |
| honesty | 576 | 0.0156 | 0.0658 | 0.0937 |
| helpfulness | 1056 | 0.0028 | 0.0468 | 0.0845 |

## Objective-label correlation

| pair | n | pearson |
|---|---|---|
| helpfulness vs honesty | 3699 | +0.507 |
| helpfulness vs instruction_following | 3615 | +0.675 |
| helpfulness vs truthfulness | 3384 | +0.470 |
| honesty vs instruction_following | 3335 | +0.503 |
| honesty vs truthfulness | 3147 | +0.602 |
| instruction_following vs truthfulness | 3075 | +0.563 |
