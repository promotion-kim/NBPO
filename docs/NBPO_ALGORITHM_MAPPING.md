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
  "dual_iterations": 40000
}
```

so no reader has to infer which of the two is on disk.

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

## What the code runs

The dual runs on a **frozen finite response pool**: four sampled learner
responses and four sampled comparator responses per prompt, judged pairwise into
the centered tensor `A[k, x, i, j]`. On that pool the inner maximiser of Eq. (17)
has a closed form — a distribution over the pooled responses (Eq. 21) — so a
dual iteration is a tensor computation costing microseconds, not a training run.
The neural policy is fit **once**, after the dual converges, by regressing the
log-ratio onto the Eq. (26) targets built from the converged `lambda` and the
opponent `nu_update`.

```
for m in 1..M:                      # frozen finite pool, no neural training
    for r in 1..R:                  # R = 1 in the manuscript (disclosed)
        nu   <- regularized opponent at the current pooled policy    (Eq. 7)
        q    <- expected payoff against nu                           (Eq. 9)
        pi   <- pi_t * exp(eta * sum_k lambda_k q_k), renormalized   (Eq. 21)
    s        <- V(pi) - d                                            (Eq. 8, 10)
    lambda   <- clamp(lambda - gamma * (s - 1/lambda), box)          (Eq. 27)

# ---- exactly one neural fit per outer stage, after the loop ----
Z_k          <- B_k - B'_k,  z_k ~ nu_update, per response pair AND objective (Eq. 24)
minimize_theta  E[ (h_t(theta) - eta * sum_k lambda_k Z_k)^2 ]                (Eq. 26)

# ---- gate ----
decode candidate on held-out monitoring prompts, judge, evaluate s_hat
if min_k s_hat_k <= 0:  reject, retain pi_t, flag empirically infeasible
else:                   promote to pi_{t+1}
```

## Where the two differ, and what that costs

| | manuscript Algorithm 1 | this code |
|---|---|---|
| policy in the dual | distribution over the full response space | distribution over the frozen pool of 4 sampled responses |
| inner solve per dual step | exact regularized best response | `R` reweightings of the closed-form pooled update (`R = 1`, residual reported) |
| neural training | not distinguished from the inner solve | exactly once per outer stage, after the dual |
| `M` | `M` exact inner solves | `M` tensor iterations on frozen judgments |

Two approximations remain, both measured rather than assumed:

1. **Finite pool.** The dual optimizes over four sampled responses per prompt,
   not the response space. `scripts/nbpo/eval_game_value.py` reports the surplus
   the *exact* finite-pool solution achieves, so the gap between it and the
   trained network is measurable rather than conflated with method error.
2. **`R = 1`.** One opponent reweighting per dual update instead of iterating the
   Eq. (21) fixed point to convergence. Both `fixed_point_residual` and
   `extra_map_residual` (the movement of one further map) are written into every
   solver artifact, so the size of this approximation is on the record.

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

Call this pipeline the **finite-pool NBPO realization**. Do not describe it as
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
| (19)-(20) `grad phi = s - 1/lambda`, `lambda_k = 1/s_k` | `nbpo_solver.py:solve_nbpo_dual` (residuals) |
| (21) raw-lambda proximal update | `nbpo_core.py:weighted_policy_update`, `nbpo_solver.py:solve_weighted_policy` |
| (22) log-ratio change `h_t` | `mnpo_scripts/mnpo_trainer.py` (`loss_type: nbpo` branch); tokenization: `mnpo_scripts/pair_tokenization.py` |
| (24) binary target `Z_k = B_k - B'_k` | `scripts/nbpo/build_nbpo_pairs.py:build_rows` |
| (26) regression loss | `mnpo_trainer.py` nbpo branch; `logp_reduction: sum` |
| (27) projected dual step | `nbpo_solver.py:solve_nbpo_dual` |
| Alg. 1 gate (lines 11-15) | `scripts/nbpo/run_nbpo_stage.py:apply_gate` |
