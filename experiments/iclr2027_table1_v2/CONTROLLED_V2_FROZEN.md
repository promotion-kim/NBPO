# Controlled-v2: frozen interpretation

Manifest: `results/iclr2027_table1_v2/controlled_v2/controlled_v2_manifest.json`
(source commit, generator and solver hashes, full construction config, measured
`rho*` calibration, SHA-256 of every result file).
Independent re-derivation: `controlled_v2_verification.json`.

## The calibration, stated correctly

`rho*` is **not** exactly 0.03 everywhere:

| alpha | rho* mean | sd | min | amplitude cap binds |
|---|---|---|---|---|
| 0.00 | 0.030000 | 7.2e-08 | 0.030000 | no |
| 0.25 | 0.030000 | 6.4e-08 | 0.030000 | no |
| 0.50 | 0.030000 | 6.7e-08 | 0.030000 | no |
| 0.75 | 0.030000 | 6.3e-08 | 0.030000 | no |
| **1.00** | **0.028334** | **2.4e-03** | **0.024746** | **2 of 5 seeds** |

At `alpha = 1` the no-clipping amplitude cap (`m <= 0.17`) binds for seeds 0 and
2, so the bisection cannot reach the target without clipping — and clipping is
refused by construction. The reported metric is **min surplus / rho\***, which
absorbs the remaining scale difference; raw min surplus at `alpha = 1` is not
comparable across seeds and is not quoted.

## Two interval routes, and why only one governs

Every paired gap is computed twice: a percentile bootstrap over seeds and an
exact paired Student-t interval, plus a distribution-free sign test.

**They disagree, and the bootstrap is the one that is wrong.** With five seeds a
percentile bootstrap resamples a five-point mean; its spread is materially too
narrow, and here it reported significance for gaps of order 0.01–0.02 that the
t-interval does not support. **Only the t verdict is quoted as established.**

A second design fact must be stated with it: at `n = 5` the smallest attainable
two-sided sign-test p is `2 * 0.5^5 = 0.0625`, so a *unanimous* sign across
seeds cannot reach the 5 % level. That is a property of running five seeds, not
of the effect.

## What is established

| claim | verdict |
|---|---|
| NBPO is **slightly below** Fixed-reference Nash at alpha 0 and 0.25 | **established** — −0.0016 [−0.0030, −0.0001] and −0.0104 [−0.0201, −0.0007] |
| NBPO vs BT-RM-Nash at alpha 0 | **indistinguishable** — −0.0014 [−0.0029, **+0.0001**]; the bootstrap called this negative and was wrong |
| NBPO is below BT-RM-Nash at alpha 0.25 | established — −0.0102 [−0.0200, −0.0005] |
| NBPO **overtakes both** at alpha >= 0.5 | **established** — vs FixedRef +0.105 [+0.081, +0.130], +0.465 [+0.390, +0.540], +1.172 [+0.872, +1.473] |
| the gap **grows with BT deviance** | **established** — OLS slope 3.42 against deviance; sign flips between alpha 0.25 and 0.50 and the intervals separate monotonically |
| NBPO and Game-KS are **statistically indistinguishable** | **established at every alpha**, on both routes — gap −0.012 to −0.001, sign split 1+/4−, 3+/2−, 2+/3−, 3+/2−, 2+/3− |
| NBPO over Game-utilitarian | **positive in 25 of 25 cells, magnitude 0.014–0.023, but established only at alpha = 0** — at alpha >= 0.25 the t-interval includes zero. **Consistent, not established at five seeds.** |
| BT-RM-Nash over BT-RM-utilitarian | same shape: positive in every cell, established only at alpha 0 and 0.25 |
| legacy R-step | **0/25 converged.** Not a valid performance arm; appears only as a failure ablation |

**Correction to an earlier statement of mine.** I previously described the
NBPO-over-Game-utilitarian gap as "real but tiny", citing the bootstrap. Under
the conservative route it is *consistent but not established* at five seeds.
Establishing it needs more seeds: ten unanimous seeds would give a sign-test
p of 0.002.

## The scientific reading, unchanged by the above

The split is by **representation**, not by aggregation rule. The three
adaptive-game rows — Nash, utilitarian, Kalai–Smorodinsky — stay together at
0.87–0.93 of `rho*` across the whole sweep. The three non-adaptive rows —
fixed-reference Nash, BT-RM-Nash, BT-RM-utilitarian — collapse together to about
−0.24. BT-RM-**Nash** is a Nash rule and it collapses exactly like its
utilitarian twin.

So this benchmark supports the **adaptive game-value representation** and does
**not** establish that Nash is superior to Game-KS. The aggregation claim belongs
to the natural SafeRLHF panel.

## Blank cells are not failures

`controlled_v2_applicability.json` records which columns exist for which method.
Cells now read a number, `NA` (mathematically inapplicable — the reference row is
not solved at all; the two exact rows have no Eq. (21) map and no dual; the
utilitarian and KS rows carry no Nash dual), or `undefined` (applies but
undefined on this instance — only `nash_welfare`, where a surplus is
nonpositive). No cell is left blank.

`n_converged` and `n_used` are separate columns, and "expected non-convergent but
used" is never counted as convergence.
