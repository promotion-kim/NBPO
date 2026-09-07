# Auditing the controlled-nontransitivity stress test

The v1 sweep reported that NBPO loses to a fixed-reference control, and to the
uniform reference itself, as intransitivity rises. That reads as a method
failure. It is not one, and this is the audit that establishes what it actually
is.

Artifacts: `results/iclr2027_table1_v2/nontransitivity_audit/`. Code:
`nontransitivity_feasibility.py` (the exact solver and its certificate),
`nontransitivity_v2.py` (the feasibility-preserving benchmark),
`audit_nontransitivity.py` (the driver), `audit_step_size_claim.py`.

## Why an exact answer is available at all

`V_{k,beta}(pi) = mean_x -beta_k logsumexp_j( log mu_xj - r_kxj / beta_k )` with
`r` linear in `pi`. `logsumexp` is convex, so `-beta * logsumexp` is concave, so
**every objective value is concave on the product of simplices**. Therefore

* `rho* = max_pi min_k s_k(pi)` is a concave max-min — a convex program;
* the Nash objective `sum_k log s_k(pi)` is concave where the surpluses are
  positive;
* the proximal Nash objective `sum_k log s_k(pi) - D(pi||pi_ref)/eta` is concave.

All three are solved directly in float64 with SLSQP on the raw simplex and
analytic gradients, and none of them relies on the alternating fixed point that
is under audit.

**Every `rho*` carries a certificate, not a convergence claim.** Concavity puts
each `s_k` below its tangent, and the resulting inner maximization over a product
of simplices is attained at a vertex per prompt, so

    rho* <= <w, s(pi)> + sum_x [ max_i g_xi - <pi_x, g_x> ],   g = sum_k w_k grad s_k(pi)

holds for every `w` on the objective simplex, in closed form. The bound is then
tightened over `w` (convex in `w`, four dimensions). The reported gap is that
bound minus the achieved `min_k s_k`.

Two independent implementations of the proximal program — one joint over `X*I`
variables, one exploiting the fact that it separates across prompts — agree to
`TV < 1e-5`, and a test pins that a converged alternating fixed point and the
direct maximizer are the same policy, which they must be: the Eq. (21) fixed
point *is* the stationarity condition of the concave program.

## Result 1 — every v1 instance is inside Assumption 1

| alpha | BT deviance | rho* (mean ± sd) | min over seeds | rho* > 0 | max certified gap | ref is max-min optimal | IR ideal (min) |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.0048 | 0.11097 ± 0.00880 | 0.10303 | 100 % | 6.6e-09 | 0 % | 0.2413 |
| 0.25 | 0.0683 | 0.07552 ± 0.00380 | 0.07132 | 100 % | 1.0e-07 | 0 % | 0.1676 |
| 0.50 | 0.2113 | 0.03543 ± 0.00151 | 0.03288 | 100 % | 1.9e-07 | 0 % | 0.0781 |
| 0.75 | 0.4453 | 0.01709 ± 0.00189 | 0.01542 | 100 % | 3.1e-07 | 0 % | 0.0401 |
| 1.00 | 0.8493 | 0.01692 ± 0.00202 | 0.01510 | 100 % | 1.7e-07 | 0 % | 0.0431 |


`rho* > 0` on **all 25 instances** (5 seeds x 5 alphas), certified to between
`6.6e-9` and `3.1e-7`, and the uniform reference is **never** the max-min optimum.
So no instance falls outside Assumption 1 and none had to be excluded. The
disagreement point is measured, not assumed: `s_k(mu) = V_k(mu) - d_k = 0`
exactly, by construction, so the reference row in every table below is a true
zero rather than a normalisation.

Game-KS is defined on every instance, and the individually rational ideal point
is comfortably positive (0.24 down to 0.040), so the degenerate case the solver
guards against -- an ideal that is positive only by float noise -- does not
arise here.

The caveat that belongs with the pass: **the feasible margin shrinks 6.5x**,
`rho* = 0.111 -> 0.017`. Cross-alpha comparisons of raw surplus are therefore
made against a moving target, which is why the rho*-matched v2 benchmark exists.

## Result 2 -- the exact objective is fine; the deployed iteration is not

### solvers — v1

| alpha | method | min surplus | / rho* | KL from ref | ‖w‖₁ | fixed-point residual | TV to exact proximal |
|---|---|---|---|---|---|---|---|
| 0.00 | A  exact global Nash | 0.1007 | 0.907 | 1.327 | — | — | 0.1565 |
| 0.00 | B  exact proximal Nash | 0.0971 | 0.875 | 0.939 | — | — | 0.0000 |
| 0.00 | C  practical (R-step map) | 0.0971 | 0.875 | 0.939 | 37.8 | 6.3e-08 | 0.0000 |
| 0.00 | C2 practical (exact inner solve) | 0.0904 | 0.815 | 0.847 | 29.6 | 3.2e-08 | 0.0414 |
| 0.00 | D  matched step | 0.0970 | 0.874 | 0.927 | 36.6 | 8.3e-02 | 0.0050 |
| 0.00 | Fixed-reference Nash | 0.0941 | 0.848 | 0.951 | 36.6 | 0.0e+00 | 0.0719 |
| 0.00 | BT-RM-Nash | 0.0947 | 0.853 | 0.961 | 6.1 | 0.0e+00 | 0.0800 |
| 0.00 | Reference (uniform) | 0.0000 | 0.000 | 0.000 | — | — | 0.5921 |
| 0.25 | A  exact global Nash | 0.0682 | 0.903 | 1.153 | — | — | 0.1278 |
| 0.25 | B  exact proximal Nash | 0.0659 | 0.873 | 0.831 | — | — | 0.0000 |
| 0.25 | C  practical (R-step map) | 0.0567 | 0.750 | 0.884 | 55.4 | 9.7e-01 | 0.0701 |
| 0.25 | C2 practical (exact inner solve) | 0.0587 | 0.778 | 0.679 | 34.1 | 1.0e-07 | 0.0679 |
| 0.25 | D  matched step | 0.0661 | 0.876 | 0.862 | 47.7 | 8.4e-01 | 0.0498 |
| 0.25 | Fixed-reference Nash | 0.0644 | 0.853 | 0.956 | 47.7 | 0.0e+00 | 0.1106 |
| 0.25 | BT-RM-Nash | 0.0641 | 0.849 | 0.962 | 9.8 | 0.0e+00 | 0.1152 |
| 0.25 | Reference (uniform) | 0.0000 | 0.000 | 0.000 | — | — | 0.5472 |
| 0.50 | A  exact global Nash | 0.0313 | 0.884 | 0.610 | — | — | 0.0493 |
| 0.50 | B  exact proximal Nash | 0.0312 | 0.881 | 0.489 | — | — | 0.0000 |
| 0.50 | C  practical (R-step map) | -0.0603 | -1.701 | 1.264 | 170.3 | 1.0e+00 | 0.5477 |
| 0.50 | C2 practical (exact inner solve) | 0.0259 | 0.731 | 0.323 | 39.4 | 4.9e-07 | 0.0769 |
| 0.50 | D  matched step | 0.0038 | 0.107 | 1.290 | 62.7 | 1.0e+00 | 0.3950 |
| 0.50 | Fixed-reference Nash | 0.0171 | 0.483 | 0.952 | 62.7 | 0.0e+00 | 0.2795 |
| 0.50 | BT-RM-Nash | 0.0168 | 0.475 | 0.954 | 14.2 | 0.0e+00 | 0.2806 |
| 0.50 | Reference (uniform) | 0.0000 | 0.000 | 0.000 | — | — | 0.3960 |
| 0.75 | A  exact global Nash | 0.0153 | 0.894 | 0.217 | — | — | 0.0138 |
| 0.75 | B  exact proximal Nash | 0.0153 | 0.893 | 0.192 | — | — | 0.0000 |
| 0.75 | C  practical (R-step map) | -0.1163 | -6.808 | 1.354 | 385.7 | 1.0e+00 | 0.6481 |
| 0.75 | C2 practical (exact inner solve) | 0.0121 | 0.707 | 0.121 | 41.7 | 1.4e-06 | 0.0461 |
| 0.75 | D  matched step | -0.0815 | -4.768 | 1.228 | 71.3 | 1.0e+00 | 0.5434 |
| 0.75 | Fixed-reference Nash | -0.0420 | -2.457 | 0.949 | 71.3 | 0.0e+00 | 0.4375 |
| 0.75 | BT-RM-Nash | -0.0429 | -2.513 | 0.963 | 17.1 | 0.0e+00 | 0.4422 |
| 0.75 | Reference (uniform) | 0.0000 | 0.000 | 0.000 | — | — | 0.2324 |
| 1.00 | A  exact global Nash | 0.0158 | 0.935 | 0.137 | — | — | 0.0074 |
| 1.00 | B  exact proximal Nash | 0.0158 | 0.936 | 0.124 | — | — | 0.0000 |
| 1.00 | C  practical (R-step map) | -0.1851 | -10.938 | 1.366 | 607.6 | 1.0e+00 | 0.6431 |
| 1.00 | C2 practical (exact inner solve) | 0.0139 | 0.821 | 0.088 | 41.6 | 1.9e-06 | 0.0270 |
| 1.00 | D  matched step | -0.1647 | -9.734 | 1.254 | 62.4 | 1.0e+00 | 0.6007 |
| 1.00 | Fixed-reference Nash | -0.0995 | -5.880 | 0.963 | 62.4 | 0.0e+00 | 0.4933 |
| 1.00 | BT-RM-Nash | -0.1001 | -5.918 | 0.969 | 14.7 | 0.0e+00 | 0.4952 |
| 1.00 | Reference (uniform) | 0.0000 | 0.000 | 0.000 | — | — | 0.1830 |


Read the `fixed-point residual` column first, because it explains every other
number in the table.

* At `alpha = 0` the deployed `R`-step solver **converges** (residual `6.3e-08`)
  and lands on the exact proximal policy to `TV = 0.0000`. The implementation is
  correct.
* At `alpha >= 0.5` its residual is **1.000** -- the largest a simplex iterate
  can have -- its weight norm reaches **608**, and it sits `TV = 0.64` from the
  policy it is supposed to be approximating. It is not converging; it is
  oscillating between vertices, because raw `lambda_k = 1/s_k` grows as surpluses
  approach zero until `eta * sum_k w_k q_k` saturates the per-prompt softmax.
* The **exact** solutions are unaffected: `A` and `B` hold 0.87-0.94 of `rho*`
  at every alpha, and they agree with each other more closely as the attainable
  set shrinks (`TV = 0.157` at alpha 0, `0.007` at alpha 1).

**The comparison the negative rested on is void.** `Fixed-reference Nash` and
`BT-RM-Nash` both have a fixed-point residual of **exactly 0** at every alpha,
because their Eq. (21) map is constant and one application lands on its fixed
point. The stress test was comparing a diverging iteration against two that
*cannot* diverge. That is a property of the update rule, not of the objective.

## Result 3 -- the repaired solver recovers the exact behaviour

`C2` is the same outer dual loop with the inner subproblem solved by concave
maximization rather than iterated (`inner_solver="exact"`):

| alpha | rho* | C (R-step) | **C2 (exact inner)** | C2 / rho* | C2 residual | C2 TV to exact proximal |
|---|---|---|---|---|---|---|
| 0.00 | 0.1110 | +0.0971 | +0.0904 | 0.815 | 3.2e-08 | 0.041 |
| 0.25 | 0.0755 | +0.0567 | +0.0587 | 0.778 | 1.0e-07 | 0.068 |
| 0.50 | 0.0354 | **-0.0603** | **+0.0259** | 0.731 | 4.9e-07 | 0.077 |
| 0.75 | 0.0171 | **-0.1163** | **+0.0121** | 0.707 | 1.4e-06 | 0.046 |
| 1.00 | 0.0169 | **-0.1851** | **+0.0139** | 0.821 | 1.9e-06 | 0.027 |

C2 is **positive at every alpha**, its residual never exceeds `1.9e-06` against
the `1e-4` gate, and it stays within `TV = 0.08` of the exact proximal policy.
Its weight norm stays in the 30-42 range instead of running to 608.

One honest asymmetry: C2 runs **120** outer dual iterations against C's **1 500**,
which is why C is marginally ahead at `alpha = 0` (0.875 vs 0.815) where both
converge. C2 wins by an order of magnitude exactly where the map stops
contracting, on a twelfth of the outer budget.

## Result 4 -- "roughly two thirds is step size" is retracted

`step_size_claim_audit.json`, computed from the recorded numbers:

| alpha | gap (native) | gap (matched step) | fraction explained by step size |
|---|---|---|---|
| 0.00 | +0.00300 | +0.00293 | **2.2 %** |
| 0.25 | -0.00777 | +0.00170 | 78.1 % (and the **sign flips**) |
| 0.50 | -0.07737 | -0.01334 | 82.8 % |
| 0.75 | -0.07434 | -0.03949 | 46.9 % |
| 1.00 | -0.08558 | -0.06520 | **23.8 %** |

It is not a constant and it is nowhere near two thirds at the alphas that matter;
at the strongest intransitivity it is 23.8 %. The claim is withdrawn.

The decomposition is meaningless anyway, for the reason in Result 2: both arms
being differenced are non-converged, so splitting their difference into "step
size" and "representation" attributes to the objective what belongs to the
iteration.

## Result 5 -- v2, the feasibility-preserving benchmark

v1 did **not** lose feasibility, so a v2 was not required by the stated trigger.
It was built anyway, because `rho*` shrinking 6.5x across the sweep confounds
every cross-alpha comparison.

v2 **adds** a cyclic component to a fixed transitive one instead of interpolating
between them:

    A_k(alpha) = T + base_k + alpha * C_k

`T` is a shared transitive direction every objective values identically -- the
feasibility witness. `base_k` carries the per-objective conflict and is fixed.
`C_k` is a per-objective circulant tournament on an **odd** number of responses,
which has exactly zero row sums, hence zero column sums, hence contributes
nothing to `d_k = V_k(mu)`. So the disagreement point provably does not move with
alpha, and the builder asserts it (it raises if `d` drifts by more than 1e-12).
Amplitudes are chosen so `|A| <= 1/2` holds outright: **nothing is clipped**, so
the reference/witness geometry is never deformed.

The matched mode bisects the weight on `T` per alpha to hold `rho*` constant.
Measured, 3 seeds:

| alpha | BT deviance | rho* (mean ± sd) | certified gap |
|---|---|---|---|
| 0.00 | 0.00027 | 0.030000 ± 5.5e-08 | 1.9e-13 |
| 0.25 | 0.02013 | 0.030000 ± 7.1e-08 | 1.1e-07 |
| 0.50 | 0.08119 | 0.030000 ± 8.2e-08 | 1.9e-07 |
| 0.75 | 0.18813 | 0.030000 ± 6.3e-08 | 3.0e-07 |
| 1.00 | 0.35024 | 0.027223 ± 2.6e-03 | 1.3e-07 |

`rho*` is held to eight decimal places for `alpha <= 0.75`; at `alpha = 1` the
no-clipping amplitude cap binds and it settles at 0.0272, which is reported
rather than forced.

At a constant feasible margin the result is as clean as it gets:

| alpha | A / rho* | B / rho* | C / rho* | C residual |
|---|---|---|---|---|
| 0.00 | 0.920 | 0.865 | 0.822 | 1.7e-08 |
| 0.25 | 0.919 | 0.880 | 0.854 | 6.3e-01 |
| 0.50 | 0.913 | 0.893 | **0.135** | 1.0e+00 |
| 0.75 | 0.914 | 0.903 | **-1.457** | 1.0e+00 |
| 1.00 | 0.918 | 0.912 | **-3.391** | 1.0e+00 |

**The exact solutions are flat in intransitivity and only the solver collapses.**
That is the sentence the whole audit exists to license.

## What this changes about the paper's claim

The stress test does *not* show that a game-valued objective fails to buy
anything as intransitivity rises. It shows that the deployed KL-proximal
iteration stops converging in exactly the regime the objective was designed for,
and that the fixed-reference and BT-RM controls were structurally immune to that
failure. With the objective solved rather than iterated, NBPO is positive at
every alpha and flat relative to what is attainable.

What the stress test still cannot do is stand in for natural data. It is a
synthetic construction; the amount of intransitivity in real preference data is a
separate question, and on the released annotations audited in
`SAFERLHF_SUPERVISION.md` the answer is that it cannot even be measured.
