# Section-7 neural realization: pre-declared protocol

Written **before** any policy was trained. Nothing below is revised after seeing
a result; if the gate fails, the failure is reported and the thresholds stay.

## What is being measured

The trainer regresses

    h_t = [log pi(y|x) - log pi(z|x)] - [log pi_t(y|x) - log pi_t(z|x)]

onto `eta * nbpo_weighted_z`. Both terms are sequence-SUM log-probabilities over
response tokens, under one tokenizer, chat template, response mask and
truncation rule, taken from the same `mnpo_scripts.precompute` path that
produced the training columns.

`nbpo_weighted_z = sum_k lambda_k z_k` is stored **unscaled**; `eta` is applied
exactly once, in the trainer (`mnpo_scripts/mnpo_trainer.py`,
`target = self.eta * nbpo_target`). Verified in
`scripts/nbpo/build_nbpo_pairs.py:135`, which forms the target from raw
`lambda` with no `eta` factor.

## The attainable floor (measured first, not after)

`h_t` is a function of `(prompt, learner pair)` only. The sampled Eq. (24)
target also depends on the drawn opponent and on two Bernoulli flips, which no
policy can see. From `target_noise_floor.py` on this pool:

| quantity | value |
|---|---|
| max attainable `r^2` | 0.1420 |
| `normalized_mse_var` floor | 0.8580 |
| pre-registered gate | < 0.90 |
| analytic vs empirical target variance | 1.013 |

So the usable band is `[0.858, 0.900]`: passing means capturing about **70% of
all predictable signal**, not 10%. A result of 0.88 is near-ceiling, and must be
read that way rather than as a weak fit. The floor is a property of the target
construction and would be the same for every method compared on this pool.

## Splits

700 training prompts / 300 held out, prompt-wise, salt `nbpo-smoke-holdout`.
The 300 split again by salt `nbpo-smoke-valtest` into 150 **validation** and 150
**test**. Validation selects; test is scored once, for the gate, and never used
to choose eta, learning rate, steps or a checkpoint.

Caveat recorded rather than hidden: the dual `lambda` was solved on all 1000
prompts, so the held-out prompts influenced the dual weights. This is the
in-pool `targets.test_prompts` holdout, not an independently generated set; it
measures whether the neural projection generalizes across prompts, not whether
the dual does.

## Sequential tuning, in this order, on validation only

1. **eta** in `{0.1, 0.3, 1.0}` at the config learning rate (5e-7), 300 steps.
   Rationale: target RMS is 6.69 nats at eta = 1, an enormous proximal step;
   the grid spans RMS 0.67 / 2.0 / 6.7. `normalized_mse_var` is scale-free in
   eta for a policy that tracks the conditional mean, so any preference the
   sweep shows is about what the policy can actually realize.
2. **learning rate** in `{2e-7, 5e-7, 1e-6}` at the selected eta.
3. **steps** in `{150, 300, 600}` at the selected eta and learning rate.

Budget: at most 9 training runs. If no configuration passes the gate on
validation, training stops there; no final seeds are run, and a failure
diagnostic is written. Thresholds and the evaluation set do not move.

## Gate (all four, on the test half, after selection on validation)

| metric | requirement |
|---|---|
| `normalized_mse_var` | < 0.90 |
| `sign_agreement` | > 0.65 |
| `pearson` | > 0 |
| `spearman` | > 0 |

Solver-side items, already met on this pool and reported separately: projected
KKT 3.55e-15, extra-map residual 3.25e-13, Eq. (26) target identity 1.78e-15,
`inner_solver = exact`, `dual_solver = root`, eta applied once.

`sign_agreement` is computed on rows where both `h` and the target are nonzero;
31.2% of targets are exactly zero because the sampled `z_k` is a difference of
two Bernoulli draws. The comparable-row count is reported alongside it.
