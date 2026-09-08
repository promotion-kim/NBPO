# Direct finite-pool concave inner solve: measured scaling

Every row below is a measurement. The 7000-prompt cases were **run**, not extrapolated from smaller ones.

Only infrastructure is varied across rows -- prompt count, pool geometry, worker processes. The per-prompt program, the `dual_tol` of 1e-06 and the declared inner stationarity requirement of 1e-04 are identical at every size; a faster number obtained by loosening either would not be a scaling result.

Tensors are SafeRLHF-shaped: two objectives whose pairwise disagreement rate is calibrated to the 24.3% measured on the released human annotations, centered preferences in [-1/2, 1/2], exactly skew-symmetric with a zero diagonal.

## Best configuration at each size

| prompts | pool | workers | solve (s) | dual evals | s / eval | proj. KKT | inner resid. | peak RSS (MB) | artifact (s) |
|---|---|---|---|---|---|---|---|---|---|
| 200 | 4+4 | 32 | 1.5 | 31 | 0.047 | 3e-14 | 3e-11 | 467 | 0.01 |
| 200 | 8+8 | 32 | 1.7 | 35 | 0.049 | 4e-15 | 5e-12 | 498 | 0.02 |
| 200 | 16+16 | 32 | 1.8 | 32 | 0.055 | 2e-14 | 1e-11 | 499 | 0.02 |
| 1000 | 4+4 | 96 | 4.4 | 33 | 0.132 | 7e-15 | 2e-11 | 499 | 0.03 |
| 1000 | 8+8 | 96 | 3.2 | 26 | 0.122 | 4e-15 | 6e-12 | 504 | 0.04 |
| 1000 | 16+16 | 96 | 4.4 | 37 | 0.119 | 3e-13 | 1e-10 | 534 | 0.07 |
| 7000 | 4+4 | 96 | 10.6 | 28 | 0.379 | 7e-10 | 4e-11 | 534 | 0.11 |
| 7000 | 8+8 | 96 | 10.9 | 27 | 0.404 | 2e-15 | 3e-11 | 544 | 0.22 |
| 7000 | 16+16 | 96 | 15.3 | 30 | 0.508 | 1e-14 | 7e-11 | 633 | 0.39 |

## Does parallelism help, and where

| prompts | pool | 1 worker(s) | 32 worker(s) | 96 worker(s) | best speedup |
|---|---|---|---|---|---|
| 200 | 4+4 | 7.6s | 1.5s | 1.6s | 5.2x |
| 200 | 8+8 | 10.0s | 1.7s | 1.9s | 5.8x |
| 200 | 16+16 | 10.7s | 1.8s | 1.9s | 6.0x |
| 1000 | 4+4 | 41.6s | 5.4s | 4.4s | 9.5x |
| 1000 | 8+8 | 42.0s | 4.1s | 3.2s | 13.2x |
| 1000 | 16+16 | 69.2s | 5.6s | 4.4s | 15.7x |
| 7000 | 4+4 | not run | 15.2s | 10.6s | -- |
| 7000 | 8+8 | not run | 15.4s | 10.9s | -- |
| 7000 | 16+16 | not run | 22.1s | 15.3s | -- |

## Convergence

* dual converged: **24 of 24** cases
* projected KKT residual <= 1e-06: **24 of 24**
* inner stationarity residual <= 1e-04: **24 of 24**

Residual distribution over all cases:

| quantity | min | median | max |
|---|---|---|---|
| projected KKT | 1.8e-15 | 1.2e-14 | 7.3e-10 |
| inverse surplus | 2.2e-15 | 1.2e-14 | 7.3e-10 |
| inner extra-map | 4.5e-12 | 2.4e-11 | 9.8e-11 |
| Eq. (26) identity | 7.8e-16 | 9.4e-16 | 1.3e-15 |

## The 7000-prompt claim, measured

At **7000 prompts, 4+4 pool, 96 workers**: one full dual solve takes **10.6 s** over 28 dual evaluations (0.379 s each), peak RSS **534 MB**, projected KKT residual **7.3e-10**, inner residual **3.5e-11**, artifact written in **0.11 s**.

The tensor itself is small -- 1.8 MB at float64 -- so memory is not the binding constraint at this scale; the per-prompt solves are.
