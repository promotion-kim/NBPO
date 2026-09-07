# v4 development_final

## Ranking

1. **P2_long** — unresolved invalid 0.0000, deterministic 0.960, identical 1.000, confident swap 0.412, order rho 0.189, template rho 0.675, max |bias| 0.404, cost 52800
2. **P2_balanced** — unresolved invalid 0.0000, deterministic 0.800, identical 1.000, confident swap 0.356, order rho 0.209, template rho 0.702, max |bias| 0.486, cost 35200

**Selected: `P2_long`**

## P2_long

| objective | unresolved inv | 1st-pass inv | determ. | identical | UF nat | raw swap | conf swap | contra | bias | tie | unresolved unc | order rho | tmpl rho | tok p50/p90/p99/max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| helpfulness | 0.0000 | 0.0000 | 1.000 | 1.000 | 0.920 | 0.486 | 0.494 | 0.514 | +0.3639 | 0.085 | 0.000 | 0.304 | 0.794 | 52/75/98/218 |
| honesty | 0.0000 | 0.0000 | 1.000 | 1.000 | 0.770 | 0.685 | 0.708 | 0.315 | +0.0368 | 0.321 | 0.000 | 0.428 | 0.675 | 61/94/131/246 |
| instruction_following | 0.0000 | 0.0000 | 1.000 | 1.000 | 0.860 | 0.414 | 0.412 | 0.586 | +0.4036 | 0.113 | 0.000 | 0.189 | 0.711 | 64/94/131/333 |
| truthfulness | 0.0000 | 0.0005 | 0.960 | 1.000 | 0.890 | 0.671 | 0.691 | 0.329 | +0.0155 | 0.229 | 0.000 | 0.404 | 0.734 | 66/107/151/241 |

## P2_balanced

| objective | unresolved inv | 1st-pass inv | determ. | identical | UF nat | raw swap | conf swap | contra | bias | tie | unresolved unc | order rho | tmpl rho | tok p50/p90/p99/max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| helpfulness | 0.0000 | 0.0000 | 1.000 | 1.000 | 0.910 | 0.503 | 0.504 | 0.497 | +0.3644 | 0.110 | 0.000 | 0.301 | 0.782 | -/-/-/- |
| honesty | 0.0000 | 0.0000 | 0.800 | 1.000 | 0.710 | 0.709 | 0.718 | 0.291 | +0.0483 | 0.438 | 0.000 | 0.411 | 0.702 | -/-/-/- |
| instruction_following | 0.0000 | 0.0000 | 0.960 | 1.000 | 0.830 | 0.354 | 0.356 | 0.646 | +0.4864 | 0.140 | 0.000 | 0.209 | 0.791 | -/-/-/- |
| truthfulness | 0.0000 | 0.0000 | 0.800 | 1.000 | 0.840 | 0.689 | 0.695 | 0.311 | +0.0608 | 0.271 | 0.000 | 0.384 | 0.768 | -/-/-/- |
