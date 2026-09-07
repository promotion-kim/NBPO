# Section 4 go/no-go, before any policy training

**Determination: NO-GO for now, on one condition, with the blocker named and
half of it already removed.** Nothing has been launched.

The gate has four conditions. Three are met; the fourth is met only on its
weaker clause, and that is what holds the decision.

## 1. Controlled feasibility is positive — **PASS**

`rho* = max_pi min_k s_k(pi) > 0` on **all 25** v1 instances (5 seeds x 5
alphas), from a direct float64 primal solver with a tangent-plane certificate;
certified gaps `6.6e-9` to `3.1e-7`. The uniform reference is never the max-min
optimum. No instance is outside Assumption 1, so none had to be excluded.

The caveat that belongs with the pass: the feasible margin **shrinks 6.5x** over
the sweep, `rho* = 0.111 -> 0.017`. Cross-alpha comparisons of raw surplus are
therefore comparing against a moving target, which is why a rho*-matched v2
benchmark now exists.

## 2. Exact NBPO behaves correctly — **PASS**

Both exact solutions reach **87–94 % of `rho*`** at every alpha on v1, and **87–92 %** on the rho*-matched v2 benchmark where the margin is held constant — flat across the whole sweep (0.920, 0.919, 0.913, 0.914, 0.918). The exact global
Nash point and the exact proximal Nash point track each other closely (TV 0.007
at alpha = 1, 0.157 at alpha = 0, the proximal term biting harder where the
attainable set is larger). The objective is not the problem.

## 3. A practical solver at fixed-point residual < 1e-4, close to the exact proximal policy — **PASS, after a fix**

**Not** with the deployed solver. The `R`-step Eq. (21) map has a fixed-point
residual of **1.000** at alpha >= 0.5 -- the largest a simplex iterate can have --
a weight norm of 608, and sits `TV = 0.64` from the exact proximal policy. Damping
does not fix it: at `R = 150, damping = 0.9` the last-step change falls to 5.4e-2
while one *undamped* map application still moves the iterate by **0.54**. The
iteration is crawling, not converging.

The subproblem does not need iterating. It is concave, and -- the fact that makes
this practical -- it **separates completely across prompts**, because
`V_k(pi) = mean_x v_{k,x}(pi_x)` and each `v_{k,x}` touches only that prompt's
row. So it is X independent I-dimensional concave programs, linear in the number
of prompts and embarrassingly parallel, not one program over X*I variables.

`solve_proximal_exact` in `mnpo_scripts/nbpo_generic.py` does that, selectable as
`inner_solver="exact"` on `solve_finite_pool` and `--inner-solver exact` on the
CLI. Measured at alpha = 1:

| | R-step map | exact inner solve | exact proximal (reference) |
|---|---|---|---|
| min surplus | −0.185 | **+0.0115** | +0.0139 |
| fixed-point residual | 1.000 | **8.2e-07** | — |
| TV to exact proximal | 0.64 | **0.027** | 0 |
| Eq. (26) identity residual | — | **6.1e-16** | — |

The identity mattered and nearly broke this: an optimizer iterate satisfies the
Eq. (26) log-ratio identity only to its own tolerance (~1e-7), and
`write_generic_solution_artifact` refuses above **1e-9**. The solver therefore
returns the Eq. (21) map *applied at* the maximizer, which makes the identity
exact by construction and moves the residual error into `extra_map_residual`,
where the pipeline already reports it.

**The default is unchanged** (`inner_solver="fixed_point"`), so no existing
artifact moves silently; a test pins that the default path is bitwise identical.
Seven tests cover the new path, including agreement with an independently written
joint solver and a linear-cost check.

## 4. The SafeRLHF GPM passes held-out validation — **PASS on the floor, UNEVALUABLE on the clause that matters**

It clears the constant-predictor floor decisively (NLL 0.557/0.525 against
0.656/0.604; balanced accuracy 0.723/0.754 against 0.500; ROC-AUC 0.801/0.846),
and the architectural guarantees hold exactly (antisymmetry residual 1.2e-7,
self-tie residual 0.0, cyclic residual 0.081 proving it is not a scalar model).

But the pre-registered clause is *"no defensible benefit over BT on any
observable nontransitive subset"*, and **that subset is empty**. SafeRLHF's
annotation graph is a near-perfect matching with **zero triangles**, so the
decisive comparison cannot be made at all -- not passed, not failed, unavailable.
On the general held-out comparison the GPM's advantage over a matched-budget BT
is −0.0039 nats on helpfulness [−0.0073, −0.0005] and nothing on harmlessness,
with no accuracy gain on either.

## Why this is a NO-GO, and what would change it

A one-seed smoke run now would train on a judge whose only demonstrated advantage
over a scalar reward model is 0.7 % of one objective's loss, in a dataset that
cannot exhibit the property the whole architecture exists to capture. It would
produce numbers, and they would not mean what the paper needs them to mean.

Three things would change the answer, in order of how much they buy:

1. **A supervision source with observable cycles.** The requirement is
   annotations where the same three responses to one prompt are pairwise
   compared. SafeRLHF and UltraFeedback both fail it, for different reasons
   (matching graph; score-induced total order). Until such a source exists, the
   nontransitivity claim has no natural-data evidence and the controlled
   benchmark carries it alone.
2. **The repaired solver run end to end** on a real pool -- pools, tensor,
   `--inner-solver exact`, target artifact, Eq. (26) pair build, one regression
   fit -- verifying that the identity, provenance and regression gates
   (normalized MSE < 0.90, sign agreement > 0.65, positive target correlation)
   hold at production scale rather than on 40 synthetic prompts.
3. ~~rho*-matched v2 results~~ — **done**. Holding `rho*` at 0.030000 (std 6e-8)
   while BT deviance rises 0.0003 -> 0.350, the exact solutions stay flat at
   0.865-0.920 of `rho*` and only the practical solver collapses (0.822, 0.854,
   0.135, -1.457, -3.391). The controlled table is no longer confounded.

Until then: no policy training, no reward-model training, no judgment bank, and
seeds 42/43/44 stay unlaunched.
