# Why the Section-7 neural realization fails, on the pre-declared budget

Six arms trained, all scored on held-out prompts under the protocol frozen in
`NEURAL_REALIZATION_PLAN.md`. **No arm passes the gate, and the reason is not the
one the sweep was designed to find.** No threshold and no evaluation set was
changed at any point.

## The result

Gate: `normalized_mse_var < 0.90`, `sign_agreement > 0.65`, Pearson and Spearman
`> 0`, on the test half after selecting on validation.

| stage | arm | val nMSE | test nMSE | test sign | test Pearson | pred RMS | target RMS |
|---|---|---|---|---|---|---|---|
| eta | eta=0.1 | 2.319 | 2.362 | 0.485 | −0.011 | 0.783 | 0.68 |
| eta | eta=0.3 | 1.121 | 1.182 | 0.490 | −0.030 | 0.807 | 2.03 |
| eta | **eta=1.0** (selected) | 1.020 | 1.026 | 0.505 | −0.026 | 0.918 | 6.78 |
| lr | **lr=2e-7** (selected) | 1.006 | — | — | — | 0.777 | 6.72 |
| lr | lr=5e-7 (= eta=1.0 arm) | 1.020 | 1.026 | 0.505 | −0.026 | 0.918 | 6.78 |
| lr | lr=1e-6 | 1.020 | — | — | — | 1.119 | 6.72 |
| *diagnostic* | lr=1e-5 | 22.31 | 18.62 | 0.517 | −0.008 | 28.41 | 6.78 |

Every value is at or above 1.0: worse than predicting the constant mean. Sign
agreement is at chance, and both correlations straddle zero. Validation and test
agree, so this is not split noise.

The `steps` stage was not run. It could not change the conclusion, for the
reason below, and running it would have spent an hour to confirm something three
independent measurements already settle.

## It is not an optimization-budget failure

The natural reading of "the policy did not fit" is "train longer or harder".
Three measurements rule that out.

**The ranking prefers not moving.** Across learning rates the best arm is the one
that moves least (`lr=2e-7`, prediction RMS 0.777). If the direction carried
signal, moving further would help.

**Moving 30x further does not help.** The out-of-grid diagnostic at `lr=1e-5`
reaches prediction RMS 28.4 against a target RMS of 6.8 -- it overshoots the
target by 4x -- and Pearson is still −0.008. The policy will move as far as
asked; the direction is uninformative.

**Every arm takes the same step regardless of what the target asks.** Prediction
RMS is 0.78 / 0.81 / 0.92 while target RMS spans 0.68 / 2.03 / 6.78, a 10x range.
The trainer shows why: pre-clip gradient norms are 2.3e4 against
`max_grad_norm: 1.0`, so gradient clipping is active on essentially every step
and the update is a fixed-size normalized step whose only free content is its
direction.

## What the direction is competing against

`h_t` is a function of `(prompt, learner pair)`. The sampled Eq. (24) target also
depends on the drawn opponent and on two Bernoulli flips, which the policy cannot
see. Measured on this pool:

| quantity | value |
|---|---|
| corr(sampled target, its own conditional mean) | +0.3775 |
| max attainable $r^2$ for any policy | 0.1425 |
| `normalized_mse_var` floor for any policy | 0.8580 |
| fraction of the regression loss that is reducible | 14.2% |

So the gate's whole usable band is `[0.858, 0.900]` and 86% of the loss is
irreducible noise by construction. The training signal is a 14% component of a
gradient whose norm is clipped by four orders of magnitude.

## The policy did not learn the predictable part either

A noisy metric could hide real learning, so `h` was scored against the
conditional mean `m` -- exactly the part of the target the policy could
reproduce -- instead of against the noisy draw:

| arm | corr(h, sampled target) | corr(h, m) | sign(h, m) |
|---|---|---|---|
| eta=0.1 | −0.011 | −0.003 | 0.412 |
| eta=0.3 | −0.030 | +0.004 | 0.435 |
| eta=1.0 | −0.026 | +0.003 | 0.448 |

(test half). `corr(h, m) <= +0.005` everywhere: the gate is not masking anything.
The check validates itself -- `corr(sampled target, m) = +0.3775` against the
independently predicted `sqrt(0.1425) = 0.3775`.

## The estimator, not the method, sets the difficulty

The pair builder offers `--target-mode rao_blackwell` alongside the manuscript's
sampled Eq. (24) construction. It replaces the two Bernoulli draws with their
probabilities and changes nothing else -- same solver artifact, same lambda, same
opponent draw, same prompts, same responses, same split salts:

| target estimator | corr with conditional mean | max attainable $r^2$ | variance |
|---|---|---|---|
| sampled (Eq. 24) | +0.3775 | 0.1425 | 44.69 |
| Rao-Blackwell | +0.9920 | 0.9840 | 6.64 |

Both are unbiased for the same quantity. One is 14% predictable and the other
98%.

**The Rao-Blackwell arm answers it, and rules the estimator out.** Same
optimizer, same clipping, same learning rate, same eta, same step count; only
the target differs:

| arm | val nMSE | test nMSE | test sign | test Pearson | test Spearman | h RMS |
|---|---|---|---|---|---|---|
| sampled, eta=1.0 | 1.020 | 1.026 | 0.505 | −0.026 | +0.001 | 0.918 |
| Rao-Blackwell | 1.070 | **1.126** | 0.535 | +0.025 | +0.055 | 0.924 |

The floor for the Rao-Blackwell target is `normalized_mse_var = 0.016`, and the
policy scores 1.126. Given a target that is 98% predictable, the projection
captures essentially none of it. The correlations are the best of any arm and
are the first that are positive on both halves, so a little signal is there --
but the estimator's noise was not what was blocking it.

The test-half MSE decomposes exactly:

```
MSE = Var(target) + h^2 - 2*cov = 5.916 + 0.854 - 0.110 = 6.661
```

The policy contributes 0.854 of variance and recovers 0.110 through correlation.
87% of the movement is noise, and since `h = 0` gives `nMSE = 1.0` by
construction, training is a net loss on held-out prompts.

Train and held-out split the same way: the training loss does fall below its
`h = 0` baseline (6.24 against 6.64, a 6% reduction) while the held-out MSE sits
above it. The fit is finding train-specific structure that does not transfer.

## Two hypotheses remain, and both are being probed

With the estimator ruled out, what is left is the optimization path itself.
Both probes hold the Rao-Blackwell target and everything else fixed:

| probe | hypothesis | change |
|---|---|---|
| `DIAGrb_clip100` | gradient clipping is the binding constraint | `max_grad_norm` 1.0 -> 100, pre-clip norms are ~9.4e3 |
| `DIAGrb_steps1200` | it is simply under-trained | 300 -> 1200 steps, i.e. 0.24 -> 0.98 epoch |

Both are diagnostics outside the pre-declared grid and neither is a selection
candidate.

## What was not done, deliberately

Final multi-seed training was **not** started: the pre-declared rule was to stop
before final seeds if the gate fails, and it failed. `\pending` cells in Tables 2
and 3 stay pending. No threshold moved, no evaluation set changed, and no arm was
selected on the test half.

## What this means for the manuscript

Section 7 currently presents the sampled construction as the training target. The
measurement above says that at this pool geometry (8+8, 1000 prompts) that
estimator delivers a 14%-predictable regression target, and that a policy trained
on it does not track the target at all. That is a claim about the *estimator*,
and it is stronger evidence than a passing gate would have been, because it comes
with the mechanism and a matched alternative.

Two decisions are for the user, not for me:

1. whether the finite-pool target for the final campaign should be the
   Rao-Blackwell estimator rather than Eq. (24);
2. whether `max_grad_norm` should scale with the target RMS, since a target whose
   scale is set by `\|lambda\|_1` (13.7 here) makes a fixed clip threshold
   equivalent to normalized SGD.
