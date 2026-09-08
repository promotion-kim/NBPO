# NBPO: manuscript Algorithm 1 vs. the finite-pool realization

The manuscript's Algorithm 1 and the code in this repository describe the same
method at **two different levels of abstraction**. Reading one as a literal
specification of the other produces a false expectation — that the 8B policy is
retrained inside every dual iteration — so this document states exactly what
each one says and where they meet.

Every artifact this pipeline writes carries the contract block

```json
{
  "implementation_type": "finite_pool_one_shot_neural_realization",
  "dual_policy_representation": "finite_response_distribution",
  "neural_fits_per_outer_stage": 1,
  "fixed_point_steps": 1,
  "dual_iterations": 40000,
  "inner_solver": "exact",
  "dual_solver": "root",
  "exactness_scope": "'exact' names ONE subproblem: the direct finite-pool concave inner solve, to a declared tolerance. The population proximal update is exact by theorem; the neural projection is approximate. The algorithm as a whole is NOT exact."
}
```

so no reader has to infer which of the two is on disk, or which solver produced
it. `inner_solver` is part of the contract because the two paths are not
interchangeable: the legacy `R`-step map converged on **0 of 25** controlled-v2
cells.

## What the manuscript states

Algorithm 1 is written at the **population** level. Its inner maximiser

```
pi*(lambda) = argmax_pi  { sum_k lambda_k s_k(pi) - (1/eta) D(pi || pi_t) }      (Eq. 17)
```

is the exact regularized best response over the *whole response space*, and the
dual iteration

```
lambda <- Pi_Lambda [ lambda - gamma ( s_hat(pi*(lambda)) - 1/lambda ) ]         (Eq. 27)
```

is stated over that exact inner solution, for `M` iterations. Read literally,
`M = 40,000` dual steps would mean 40,000 exact policy optimizations.

## Three levels, and what is exact at each

Conflating these is the single easiest way to overclaim, so they are named
separately and used consistently everywhere in the codebase and the manuscript.

| level | status | what it means |
|---|---|---|
| **exact population proximal update** | *exact, by theorem* | Eq. (17) over the whole response space. The population theorem is unchanged by anything below. |
| **direct finite-pool concave inner solve** | *solved numerically, to a declared tolerance* | On the frozen pool the Eq. (18) subproblem is concave and **separates across prompts**, so it is solved directly rather than iterated. Declared tolerances: dual `1e-6`, inner stationarity `1e-4`. |
| **neural projection** | *approximate* | The finite-pool log-ratio target is regressed into the 8B policy. This is the approximation the method actually carries. |

**The algorithm as a whole is not exact.** `inner_solver: exact` names the middle
row only — one subproblem, solved instead of iterated. `mnpo_scripts/final_run_validator.py`
states this in its own report so the distinction survives into every artifact.

## What the code runs

The dual runs on a **frozen finite response pool**: sampled learner and
comparator responses per prompt, judged pairwise into the centered tensor
`A[k, x, i, j]`.

The key structural fact is that `V_{k,beta}(pi) = mean_x v_{k,x}(pi_x)` and each
`v_{k,x}` depends on `pi` **only through that prompt's row**. So the inner
problem is not one program over `X * I` variables but `X` independent concave
programs over the `I`-simplex — which is why it scales, and why it parallelizes
without approximation.

```
# ---- 1. finite response-pool preference tensors ----
A[k,x,i,j]   <- judge the pooled responses pairwise, centered            (Eq. 2)

# ---- 2-4. dual loop on the frozen pool: no neural training ----
solve for lambda:  s_k(pi(lambda)) = 1 / lambda_k                       (Eq. 19-20)
    where, for each trial lambda:
      # 2. prompt-separable concave inner solve, to a declared tolerance
      for each prompt x, independently and in parallel:
          pi_x <- argmax_p  eta * sum_k lambda_k v_{k,x}(p) - KL(p || pi_t,x)
      # the Eq. (21) map is applied AT that maximizer, so the Eq. (26)
      # identity holds to float64 by construction (residual ~1e-15)
      pi   <- pi_t * exp(eta * sum_k lambda_k q_k(pi_hat)), renormalized  (Eq. 21)
      s    <- V(pi) - d                                                  (Eq. 8, 10)
    # 3. the dual is a K-dimensional ROOT problem, not a descent
    lambda <- root of (s - 1/lambda), in log-lambda coordinates          (Eq. 27)
    stop on the projected KKT residual, not on an iteration count

# 4. the finite-pool target policy
pi*        <- the converged pooled policy;  target = log pi* - log pi_t

# ---- 5. exactly one neural fit per outer stage ----
Z_k        <- B_k - B'_k,  z_k ~ nu_update, per response pair AND objective (Eq. 24)
minimize_theta  E[ (h_t(theta) - eta * sum_k lambda_k Z_k)^2 ]            (Eq. 26)

# ---- 6. evaluate and promote ----
decode candidate on held-out monitoring prompts, score, evaluate s_hat
if min_k s_hat_k <= 0:  reject, retain pi_t, flag empirically infeasible
else:                   promote to pi_{t+1}
```

Measured cost of steps 2-4 at production scale (`results/iclr2027_table1_v2/solver_scaling/`):
7000 prompts, 4+4 pool, 96 workers — **10.6 s** for a complete dual solve over 28
dual evaluations, peak RSS 534 MB, projected KKT `7.3e-10`. At 16+16, 15.3 s.
All 24 measured cases converged.

## Where the two differ, and what that costs

| | manuscript Algorithm 1 | this code |
|---|---|---|
| policy in the dual | distribution over the full response space | distribution over the frozen pool |
| inner solve per dual step | exact regularized best response | **direct concave solve of the prompt-separable subproblem**, inner residual `<= 1e-4` (measured `4.5e-12`-`9.8e-11`) |
| dual iteration | `M` projected subgradient steps | root solve on `s = 1/lambda`, stopped on the projected KKT residual (`<= 1e-6`, measured `<= 7.3e-10`) |
| neural training | not distinguished from the inner solve | exactly once per outer stage, after the dual |

**One** approximation now remains in the finite-pool leg, and it is measured
rather than assumed:

* **Finite pool.** The dual optimizes over the sampled responses per prompt, not
  the response space. `scripts/nbpo/eval_game_value.py` reports the surplus the
  exact finite-pool solution achieves, so the gap between it and the trained
  network is measurable rather than conflated with method error.

The second approximation this document used to list — `R = 1`, one opponent
reweighting instead of iterating the Eq. (21) fixed point — **is gone from the
final path**. See the appendix note below for what happened to it.

## Appendix: the R-step alternating map, and why it is an ablation

The original implementation iterated the Eq. (21) map `R` times per dual step.
That map is *exactly* mirror ascent at step size 1 on the same concave objective,
so it contracts only while `eta * sum_k lambda_k q_k` stays small. Raw Nash
multipliers are `lambda_k = 1/s_k` and therefore diverge as a surplus approaches
zero, at which point the exponent saturates the per-prompt softmax and the
iteration oscillates between vertices instead of converging.

Measured on the controlled-v2 benchmark
(`results/iclr2027_table1_v2/controlled_v2/`):

| | `R`-step map | direct inner solve |
|---|---|---|
| cells converged | **0 / 25** | 25 / 25 |
| fixed-point residual at `alpha = 1` | **1.000** | `<= 1e-4` |
| normalized min surplus at `alpha = 1` | **-3.373** | **+0.931** |
| weight norm | 608 | 30-42 |

Damping does not repair it: at `R = 150, damping = 0.9` the last-step change falls
to `5.4e-2` while one *undamped* application still moves the iterate by `0.54` —
the iteration is crawling, not converging.

**The `R`-step map may appear in the manuscript only as an approximation and
failure ablation, never as a performance arm.** `final_run_validator.py` refuses
any final configuration that names it, and refuses a final run that consumes a
solver artifact produced by it.

## One tokenization, two paths

Eq. (22) subtracts the proximal centre's sequence log-probability from the
current policy's:

```
h_t = (log pi(y|x) - log pi(y'|x)) - (log pi_t(y|x) - log pi_t(y'|x))
```

The two terms come from different code — `mnpo_scripts.precompute` scores `pi_t`
offline, `scripts.simpo_trainer` scores `pi` online — so the subtraction is only
meaningful if both score **exactly the same token ids under exactly the same
attention and label masks**. Both now call one implementation,
`mnpo_scripts/pair_tokenization.py`, which:

* joint-tokenizes `prompt + response` and backs the boundary up one token when
  the tokenizer merged across it;
* adds BOS at most once and EOS at most once;
* truncates the prompt first, then the response, with identical bounds;
* keeps every valid prompt token in the attention mask and masks the prompt in
  `labels` only;
* treats chosen and rejected symmetrically.

The settings that determine the token sequence are hashed into
`tokenization_config_sha256` and recorded in `precompute_meta.json` and
`run_config.yaml`; training refuses to start if they differ. The invariant test
`test_h_t_of_pi_t_is_zero_before_any_optimizer_step` scores one frozen model
through both production paths with a tokenizer that merges across the boundary
and asserts `h_t = 0` exactly.

## Terminology

Call this pipeline the **finite-pool NBPO realization**, and call its inner
solve a **direct finite-pool concave inner solve**. Do not call the algorithm
"exact" without that qualifier: only the finite-pool inner subproblem is solved
directly, and the neural projection remains approximate. Do not describe it as
"paper-exact Algorithm 1": that claim is only true if the manuscript's pseudocode
is rewritten to match the loop above, or the code is changed to solve the neural
inner problem at every dual iteration.

## Equation → function map

| equation | function |
|---|---|
| (2) centered payoff `A_k = P_k - 1/2` | `scripts/nbpo/build_preference_tensor.py:fill_policy_tensor` |
| (1)-(2) skew reference tensor | `build_preference_tensor.py:fill_reference_tensor` |
| (6) margins `r_{k,pi}` | `mnpo_scripts/nbpo_core.py:compute_margins` |
| (7) regularized opponent `nu*` | `nbpo_core.py:compute_regularized_opponent` |
| (8) soft-min game value `V_{k,beta}` | `nbpo_core.py:compute_regularized_game_value` |
| (9) objective gradient `q_{k,pi}` | `nbpo_core.py:compute_objective_gradient` |
| (10) disagreement point `d_k`, surplus `s_k` | `nbpo_core.py:compute_disagreement_point`, `compute_surplus` |
| (11) Nash welfare `sum_k log s_k` | `scripts/nbpo/eval_game_value.py:evaluate_game_value` |
| (17) dual objective `phi_t(lambda)` | `mnpo_scripts/nbpo_solver.py:dual_objective_phi` |
| (18) **prompt-separable concave inner solve** (final path) | `mnpo_scripts/nbpo_generic.py:solve_proximal_exact` |
| (19)-(20) `grad phi = s - 1/lambda`, `lambda_k = 1/s_k` | `nbpo_generic.py:_solve_nash_dual_by_root` (final), `nbpo_solver.py:solve_nbpo_dual` (legacy) |
| (21) raw-lambda proximal update | `nbpo_generic.py:exp_update`; `nbpo_core.py:weighted_policy_update` (legacy) |
| (21) `R`-step alternating map | `nbpo_generic.py:solve_proximal` — **ablation only** |
| (22) log-ratio change `h_t` | `mnpo_scripts/mnpo_trainer.py` (`loss_type: nbpo` branch); tokenization: `mnpo_scripts/pair_tokenization.py` |
| (24) binary target `Z_k = B_k - B'_k` | `scripts/nbpo/build_nbpo_pairs.py:build_rows` |
| (26) regression loss | `mnpo_trainer.py` nbpo branch; `logp_reduction: sum` |
| (27) projected dual step | `nbpo_generic.py:solve_finite_pool` (`dual_solver="root"`, final); `nbpo_solver.py:solve_nbpo_dual` (legacy subgradient) |
| final-config admission | `mnpo_scripts/final_run_validator.py:validate_final_config` |
| Alg. 1 gate (lines 11-15) | `scripts/nbpo/run_nbpo_stage.py:apply_gate` |
