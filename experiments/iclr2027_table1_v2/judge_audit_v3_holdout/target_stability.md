# Downstream target stability

Gates: **FAIL**

- FAIL: order/NBPO: target Pearson 0.869 < 0.9
- FAIL: order/NBPO: target sign agreement 0.736 < 0.85
- FAIL: order/Fixed-reference Nash: target Pearson 0.870 < 0.9
- FAIL: order/Fixed-reference Nash: target sign agreement 0.771 < 0.85
- FAIL: template/NBPO: not measured
- FAIL: template/Fixed-reference Nash: not measured

## order split — 24 prompts complete in both halves

| method | Delta r | Delta rho | target r | target rho | sign agree | TV median | TV p90 | rank agree |
|---|---|---|---|---|---|---|---|---|
| NBPO | 0.689 | 0.627 | **0.869** | 0.664 | **0.736** | 0.2237 | 0.7045 | 0.517 |
| Fixed-reference Nash | 0.689 | 0.627 | **0.870** | 0.786 | **0.771** | 0.1900 | 0.4637 | 0.450 |

Per-objective differences (half A minus half B):

- NBPO: value instruction_following +0.1281, truthfulness +0.0464, honesty +0.0707, helpfulness +0.1241; surplus instruction_following +0.1335, truthfulness +0.0451, honesty +0.0680, helpfulness +0.1204
- Fixed-reference Nash: value instruction_following +0.0863, truthfulness +0.0742, honesty +0.0255, helpfulness +0.0566; surplus instruction_following +0.0863, truthfulness +0.0742, honesty +0.0255, helpfulness +0.0566

## template split

no complete prompt in both halves
