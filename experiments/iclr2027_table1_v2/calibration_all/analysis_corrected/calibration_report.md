# Judge protocol calibration

Selection uses calibration and reliability metrics only. No NBPO, fixed-reference, Game-KS or policy quantity was consulted, and the ordering below was fixed before the numbers were seen.

## Ranking

1. **P3** — max invalid rate 0.000, min deterministic-degradation accuracy 0.480, min clear-control accuracy 0.640, identical confident-tie 1.000, max |position bias| 0.048, confident swap 0.669, split-half rho 0.437, cost 4800 renderings
2. **P0** — max invalid rate 0.000, min deterministic-degradation accuracy 0.140, min clear-control accuracy 0.527, identical confident-tie 1.000, max |position bias| 0.254, confident swap 0.564, split-half rho 0.838, cost 4800 renderings
3. **P1** — max invalid rate 0.000, min deterministic-degradation accuracy 0.000, min clear-control accuracy 0.540, identical confident-tie 1.000, max |position bias| 0.289, confident swap 0.530, split-half rho 0.870, cost 4800 renderings
4. **P2** — max invalid rate 0.035 **(GATE 1 FAIL)**, min deterministic-degradation accuracy 0.960, min clear-control accuracy 0.833, identical confident-tie 1.000, max |position bias| 0.062, confident swap 0.891, split-half rho 0.890, cost 4800 renderings

**NO ADMISSIBLE CANDIDATE.** Every protocol fails at least one known-answer prerequisite (invalid rate < 0.2%, deterministic-degradation accuracy >= 0.90 per objective). The ranking above is reported for diagnosis only and must not be read as a selection -- picking the least-bad protocol here is precisely how an inadmissible one gets frozen.

## Per objective

### P0

| objective | clear acc | natural | degrade | ident. tie | pos. bias | confident swap | split-half rho | mean \|D\| | tie mass | entropy |
|---|---|---|---|---|---|---|---|---|---|---|
| helpfulness | 0.927 | 0.890 | 1.000 | 1.000 | -0.0192 | 0.952 | 0.984 | 0.3496 | 0.261 | 0.000 |
| honesty | 0.527 | 0.720 | 0.140 | 1.000 | -0.2542 | 0.564 | 0.838 | 0.2013 | 0.318 | 0.000 |
| instruction_following | 0.893 | 0.880 | 0.920 | 1.000 | -0.0950 | 0.831 | 0.948 | 0.2842 | 0.345 | 0.000 |
| truthfulness | 0.713 | 0.890 | 0.360 | 1.000 | -0.1617 | 0.704 | 0.859 | 0.2250 | 0.377 | 0.000 |

### P1

| objective | clear acc | natural | degrade | ident. tie | pos. bias | confident swap | split-half rho | mean \|D\| | tie mass | entropy |
|---|---|---|---|---|---|---|---|---|---|---|
| helpfulness | 0.947 | 0.920 | 1.000 | 1.000 | -0.0263 | 0.937 | 0.963 | 0.3224 | 0.257 | 0.087 |
| honesty | 0.540 | 0.810 | 0.000 | 1.000 | -0.2894 | 0.530 | 0.870 | 0.1719 | 0.309 | 0.212 |
| instruction_following | 0.920 | 0.880 | 1.000 | 1.000 | -0.1231 | 0.745 | 0.941 | 0.2257 | 0.339 | 0.200 |
| truthfulness | 0.653 | 0.860 | 0.240 | 1.000 | -0.1447 | 0.748 | 0.882 | 0.2001 | 0.381 | 0.214 |

### P2

| objective | clear acc | natural | degrade | ident. tie | pos. bias | confident swap | split-half rho | mean \|D\| | tie mass | entropy |
|---|---|---|---|---|---|---|---|---|---|---|
| helpfulness | 0.947 | 0.920 | 1.000 | 1.000 | +0.0125 | 0.973 | 0.979 | 0.3563 | 0.258 | 0.000 |
| honesty | 0.833 | 0.750 | 1.000 | 1.000 | +0.0133 | 0.927 | 0.934 | 0.3183 | 0.278 | 0.000 |
| instruction_following | 0.899 | 0.848 | 1.000 | 1.000 | +0.0616 | 0.917 | 0.956 | 0.3243 | 0.277 | 0.000 |
| truthfulness | 0.909 | 0.882 | 0.960 | 1.000 | -0.0086 | 0.891 | 0.890 | 0.3009 | 0.305 | 0.000 |

### P3

| objective | clear acc | natural | degrade | ident. tie | pos. bias | confident swap | split-half rho | mean \|D\| | tie mass | entropy |
|---|---|---|---|---|---|---|---|---|---|---|
| helpfulness | 0.927 | 0.890 | 1.000 | 1.000 | -0.0242 | 0.947 | 0.933 | 0.3396 | 0.254 | 0.000 |
| honesty | 0.640 | 0.720 | 0.480 | 1.000 | +0.0208 | 0.669 | 0.437 | 0.1946 | 0.291 | 0.000 |
| instruction_following | 0.920 | 0.880 | 1.000 | 1.000 | -0.0300 | 0.908 | 0.853 | 0.3033 | 0.298 | 0.000 |
| truthfulness | 0.800 | 0.880 | 0.640 | 1.000 | +0.0475 | 0.777 | 0.646 | 0.2288 | 0.349 | 0.000 |
