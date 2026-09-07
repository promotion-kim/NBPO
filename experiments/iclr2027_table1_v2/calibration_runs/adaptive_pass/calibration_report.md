# Judge protocol calibration

Selection uses calibration and reliability metrics only. No NBPO, fixed-reference, Game-KS or policy quantity was consulted, and the ordering below was fixed before the numbers were seen.

## Ranking

1. **P2** — min clear-control accuracy 0.852, identical confident-tie 1.000, max |position bias| 0.035, confident swap 0.953, split-half rho 0.326, cost 1754 renderings
2. **P1** — min clear-control accuracy 0.540, identical confident-tie 1.000, max |position bias| 0.230, confident swap 0.608, split-half rho 0.630, cost 2496 renderings
3. **P0** — min clear-control accuracy 0.460, identical confident-tie 1.000, max |position bias| 0.275, confident swap 0.548, split-half rho n/a, cost 1600 renderings

**Selected: `P2`**

## Per objective

### P0

| objective | clear acc | natural | degrade | ident. tie | pos. bias | confident swap | split-half rho | mean \|D\| | tie mass | entropy |
|---|---|---|---|---|---|---|---|---|---|---|
| helpfulness | 0.893 | 0.850 | 0.980 | 1.000 | -0.0250 | 0.939 | n/a | 0.3475 | 0.260 | 0.000 |
| honesty | 0.460 | 0.660 | 0.060 | 1.000 | -0.2750 | 0.548 | n/a | 0.1975 | 0.325 | 0.000 |
| instruction_following | 0.807 | 0.830 | 0.760 | 1.000 | -0.0950 | 0.862 | n/a | 0.2850 | 0.350 | 0.000 |
| truthfulness | 0.573 | 0.750 | 0.220 | 1.000 | -0.2150 | 0.638 | n/a | 0.2075 | 0.375 | 0.000 |

### P1

| objective | clear acc | natural | degrade | ident. tie | pos. bias | confident swap | split-half rho | mean \|D\| | tie mass | entropy |
|---|---|---|---|---|---|---|---|---|---|---|
| helpfulness | 0.940 | 0.910 | 1.000 | 1.000 | -0.0207 | 0.936 | 0.630 | 0.3240 | 0.258 | 0.091 |
| honesty | 0.540 | 0.800 | 0.020 | 1.000 | -0.2300 | 0.608 | 0.756 | 0.1929 | 0.303 | 0.216 |
| instruction_following | 0.907 | 0.880 | 0.960 | 1.000 | -0.0942 | 0.800 | 0.859 | 0.2312 | 0.347 | 0.216 |
| truthfulness | 0.687 | 0.860 | 0.340 | 1.000 | -0.1157 | 0.769 | 0.751 | 0.2055 | 0.392 | 0.226 |

### P2

| objective | clear acc | natural | degrade | ident. tie | pos. bias | confident swap | split-half rho | mean \|D\| | tie mass | entropy |
|---|---|---|---|---|---|---|---|---|---|---|
| helpfulness | 0.953 | 0.929 | 1.000 | 1.000 | +0.0138 | 0.993 | 0.680 | 0.3612 | 0.260 | 0.000 |
| honesty | 0.852 | 0.772 | 1.000 | 1.000 | +0.0286 | 0.961 | 0.326 | 0.3294 | 0.296 | 0.000 |
| instruction_following | 0.902 | 0.849 | 1.000 | 1.000 | +0.0350 | 0.953 | 0.654 | 0.3323 | 0.289 | 0.000 |
| truthfulness | 0.858 | 0.847 | 0.875 | 1.000 | -0.0044 | 0.953 | 0.642 | 0.3051 | 0.347 | 0.000 |
