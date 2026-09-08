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

`h_t` is a function of `(prompt, learner pair)` only. The sampled ESTIMATOR of
the Eq. (24) target also depends on the drawn opponent and on two Bernoulli flips, which no
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

## Implementation-contract checks, measured on the trained arms

`eta` applied exactly once, verified from the trainer's own logged target rather
than by reading the code: mean `|target|` over 30 logged steps is

| eta | mean `|target|` | ratio to eta=0.1 |
|---|---|---|
| 0.1 | 0.4925 | 1.000 |
| 0.3 | 1.4774 | 2.9998 |
| 1.0 | 4.9245 | 9.9990 |

Linear in `eta` to four figures. A second application anywhere in the chain
would give ratios of 9 and 100.

Sequence-sum reduction: `precompute_meta.json` records `logp_reduction: sum`,
and `validate_nbpo_args` refuses the `nbpo` branch otherwise. Provenance: the
run config carries the pair-artifact, solver-artifact, precompute-manifest and
tokenization-config hashes of this stage, checked before any weights load.

The sampled target is not the deterministic finite-pool target and is not
treated as one. `nbpo_weighted_z` takes values in
`{0, ±lambda_1, ±lambda_2, ±(lambda_1+lambda_2)}` -- a difference of two
Bernoulli draws against a sampled opponent -- with RMS 6.69, while its own
conditional mean has RMS 2.54. The name says "target"; the quantity is a
high-variance unbiased draw whose predictable part is 14% of its variance.

## Declared BEFORE running it: the Rao-Blackwell diagnostic

The eta stage failed the gate on all three arms, and the conditional-mean check
showed the policy learned nothing -- not even the predictable part
(`corr(h, m) <= +0.005` on every test half). The open question is whether the
neural projection can fit an Eq. (26) target *at all*, or whether the sampled
estimator's noise makes the regression untrainable at this budget.

The pair builder already offers `--target-mode rao_blackwell`, described there as
a labeled variant of the sampled estimator the pair builder uses for Eq. (24). It replaces
the two Bernoulli draws with their conditional probabilities and keeps everything
else -- same solver artifact, same lambda, same opponent draw, same prompts,
same responses, same split salts. Measured on this pool:

| target estimator | corr with the conditional mean | max attainable $r^2$ | variance |
|---|---|---|---|
| sampled estimator of Eq. (24) | +0.3775 | 0.1425 | 44.69 |
| Rao-Blackwell | +0.9920 | 0.9840 | 6.64 |

So a policy could in principle explain 98% of the Rao-Blackwell target and only
14% of the sampled one. Running one arm on it isolates "the projection cannot
fit" from "the target is too noisy to fit at this budget".

**Status of this arm: diagnostic, not a selection candidate.** It trains on a
different target and therefore is not comparable to the pre-registered arms under
the pre-registered gate; it does not enter the eta/LR/steps selection, and it
does not change the gate thresholds or the evaluation set. Its only job is to
answer the structural question the failure diagnostic has to answer.

The `lr1e-5` arm is diagnostic on the same footing and for the same reason: it
sits outside the pre-declared `{2e-7, 5e-7, 1e-6}` grid and answers whether the
failure is an optimization-budget failure.
