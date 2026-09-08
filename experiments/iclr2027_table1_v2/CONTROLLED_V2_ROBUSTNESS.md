# Controlled-v2 robustness: a second cycle family, and a temperature sweep

The primary result (`CONTROLLED_V2_FROZEN.md`) uses one construction. Its
`alpha`-invariance comes from a number-theoretic property — a circulant
tournament on an **odd** number of responses has exactly zero row sums, so the
cyclic component contributes nothing to the disagreement point. If the finding
only holds there, it is a property of that construction and not of the method.

## The independent family

    A_k(alpha) = T + base_k + alpha * C_k,      C = N (R - R^T) N

`R - R^T` is a random skew matrix and `N = I - U(U^T U)^+ U^T` is the symmetric
projector annihilating both the uniform reference `mu` and a designated witness
policy `pi_bar` (built from the shared transitive direction). Conjugating a skew
matrix by a symmetric projector keeps it skew, so `C^T = -C` and `diag(C) = 0`
hold exactly, and

* `mu^T C = -(C mu)^T = 0`, so the payoff rows at the reference are untouched and
  **the disagreement point does not move with alpha** — the same invariance, from
  a projection rather than from circulant arithmetic. The builder raises if `d`
  drifts by more than `1e-12`, and it does not;
* `C pi_bar = 0`, so the witness keeps its surplus and the problem stays feasible.

No clipping, exact skew symmetry, zero diagonal, all asserted per instance.
`rho*` needs no bisection here — it is already near-constant, drifting only from
0.0297 to 0.0280 across the sweep — so it is reported as measured and the
normalized metric absorbs the rest.

## Result: the trend replicates, the magnitude does not

5 seeds, `beta = 0.25`, normalized min surplus (`min s / rho*`):

| alpha | BT dev. | NBPO | Game-KS | Game-util | Fixed-ref | BT-RM-Nash | BT-RM-util |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.0001 | +0.923 | +0.921 | +0.921 | +0.923 | +0.923 | +0.921 |
| 0.25 | 0.0100 | +0.934 | +0.929 | +0.933 | +0.932 | +0.932 | +0.930 |
| 0.50 | 0.0403 | +0.944 | +0.940 | +0.944 | +0.925 | +0.925 | +0.923 |
| 0.75 | 0.0931 | +0.951 | +0.948 | +0.951 | +0.896 | +0.896 | +0.894 |
| 1.00 | 0.1737 | **+0.956** | +0.954 | +0.956 | **+0.845** | **+0.845** | **+0.843** |

Paired gaps, conservative Student-t intervals over 5 seeds:

| alpha | NBPO − Fixed-ref | verdict |
|---|---|---|
| 0.00 | +0.0000 [−0.0003, +0.0004] | indistinguishable |
| 0.25 | +0.0026 [−0.0017, +0.0069] | indistinguishable |
| 0.50 | +0.0191 [+0.0109, +0.0272] | **positive** |
| 0.75 | +0.0555 [+0.0363, +0.0747] | **positive** |
| 1.00 | +0.1114 [+0.0714, +0.1514] | **positive** |

NBPO − BT-RM-Nash is the same to three decimals at every alpha.

**The qualitative finding replicates.** The three adaptive-game rows stay together
and drift *upward*; the three non-adaptive rows stay together and fall. The split
is again by representation, not by aggregation rule.

**The magnitude does not, and the claim must be scoped accordingly.** The gap at
`alpha = 1` is **+0.111** here against **+1.172** on the circulant family — an
order of magnitude smaller. The proximate reason is visible in the table: this
family's BT deviance only reaches **0.174** against **0.351**, because projecting
out two directions leaves a weaker cyclic component at the same amplitude. The
paper should therefore claim that *adaptive game representations remain robust as
nontransitivity rises*, and should not attach the circulant family's effect size
to that claim as if it were construction-independent.

## Temperature robustness

3 seeds, projected family, `alpha` in {0, 0.5, 1.0}. Gap is NBPO − Fixed-ref:

| beta | alpha=0 | alpha=0.5 | alpha=1.0 |
|---|---|---|---|
| **0.10** | −0.0003 [−0.0010, +0.0004] | +0.0482 [+0.0131, +0.0834] | **+0.2381 [+0.0880, +0.3883]** |
| **0.25** | +0.0000 [−0.0003, +0.0004] | +0.0191 [+0.0109, +0.0272] | **+0.1114 [+0.0714, +0.1514]** |
| **0.50** | −0.0001 [−0.0003, +0.0001] | +0.0074 [−0.0067, +0.0216] | **+0.0448 [+0.0165, +0.0730]** |

The effect survives at every temperature tested and is **monotone decreasing in
beta** — which is what the theory predicts, and is worth stating as a
consistency check rather than as a separate finding: `beta -> infinity` sends the
regularized opponent to `mu`, i.e. exactly the fixed-reference control, so the
adaptive advantage *must* vanish as `beta` grows. It does, smoothly, by a factor
of five from 0.10 to 0.50.

At `beta = 0.5, alpha = 0.5` the gap is no longer distinguishable from zero,
consistent with the same trend.

## Convergence

Every method converged in 25 of 25 cells in every run, except the legacy
`R`-step map at **0 of 25**, as in the primary family. Its non-convergence is the
ablation's result and it is never a performance arm.

## Artifacts

`controlled_v2_{raw,per_seed,summary,runtime}_projected.*`,
`..._projected_beta0.1.*`, `..._projected_beta0.5.*`,
`controlled_v2_gaps_projected.json`,
`controlled_v2_gap_figure_projected.{pdf,png}`.
