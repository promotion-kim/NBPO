# Corrections to the Section-7 failure report

The measurements in `NEURAL_REALIZATION_FAILURE.md` stand; several of the
*interpretations* attached to them went further than the data. Nothing below
deletes a number, a checkpoint or a failure record -- each item states what was
written, what is actually supported, and why the difference matters.

## 1. "The estimator is ruled out" was too strong

**Was:** "The Rao-Blackwell arm rules out the estimator, not the projection."

**Is:** reducing estimator noise alone did not clear the gate *under this trainer
and this 300-step budget*. That is the observation. It does not establish that
sampled noise plays no role, and it certainly does not establish that the neural
projection is structurally incapable -- the tiny deterministic fit later showed
the opposite, reaching an 87% loss reduction on the same target family.

**Why it matters:** "estimator ruled out" would have closed a line of enquiry
that is still open. The honest scope is one trainer, one budget.

## 2. "98.4% predictable" needed its definition stated

**Was:** a single number with no formula.

**Is:** two different quantities, now both computed:

| quantity | sampled | partial RB | canonical |
|---|---|---|---|
| `corr(T, m)^2` | 0.1425 | 0.9840 | 1.0000 |
| conditional-variance ratio `Var(m) / (Var(m) + E[Var(T \| x,pair)])` | 0.1420 | 0.9833 | 1.0000 |

They agree closely here but are not the same computation. Both describe how much
of the *target's* randomness is removable. Neither says an LLM will find the
remaining signal easy to predict on a new prompt -- that is a generalization
question, measured separately.

## 3. "87% of the movement is noise" was not a variance decomposition

**Was:** "the policy contributes 0.854 of variance and recovers 0.110 through
correlation; 87% of the movement is noise."

**Is:** `E[h^2] - 2 E[hT] = 0.744 > 0` means the increase in squared prediction
magnitude was not offset by alignment with the target. It is an accounting
identity about the MSE, not a causal decomposition of `h` into signal and noise,
and it licenses no statement about what fraction of the policy's movement is
"noise".

**Also corrected:** the decomposition was written as
`MSE = Var(T) + E[h^2] - 2 Cov`, which is exact only at `mean(T) = 0`. Measured
`mean(T)` is +0.040 (validation) and +0.089 (test), and that formula is off by
up to 1.8e-2. The two exact identities are now reported and both close to ~1e-14:

```
MSE = E[T^2] + E[h^2] - 2 E[hT]
MSE = Var(T) + Var(h) - 2 Cov(h,T) + (mean(h) - mean(T))^2
```

For the same reason, "h = 0 scores exactly 1.0 on a variance-normalized metric"
is exact only at zero target mean. `nMSE_zero` (normalizing by `E[T^2]`) is now
reported alongside `nMSE_var`; on these arms they differ in the fourth decimal,
so no conclusion turns on the choice.

## 4. "Fits train then overfits" was not established

**Was:** "the fit is finding train-specific structure that does not transfer."

**Is:** the training loss fell ~6% below its own `h = 0` baseline while held-out
MSE stayed above it. A 6% reduction is not a well-fit training set. Underfitting
on train and a generalization gap can coexist, and the 300-step arm shows signs
of both. The comparison is also imperfect in kind: training loss is an average
over online minibatches during optimization, while the held-out number is a
final-checkpoint evaluation, so they are not two readings of one quantity.

## 5. "Gradient clipping is the mechanism" was over-attributed

**Was:** pre-clip gradient norms of 2.3e4 against `max_grad_norm: 1.0` "make
every update a fixed-size normalized step".

**Is:** global-norm clipping rescales but preserves the gradient *direction*, and
the realized update is set by the learning rate, the optimizer state and
Adafactor's internal scaling, none of which were measured at the time. The
`clip100` probe then moved `max_grad_norm` by 100x and changed held-out nMSE from
1.126 to 1.093 -- a small shift, not a fix. Clipping interacts with the
optimizer; it is not the cause.

## 6. A dtype claim that was simply wrong

**Was:** "training loads fp32 with bf16 autocast while precompute loads pure
bf16 -- the prime suspect for the zero-step mismatch."

**Is:** the base checkpoint's `config.json` already declares
`torch_dtype: bfloat16`, so both paths load bf16. Setting it explicitly produced
a run identical to six decimal places. The zero-step mismatch is real but its
cause is not dtype.

## 7. What the failed checkpoints are

**Was:** "the failed policy checkpoints back no reported number."

**Is:** they are exactly the artifacts behind every reported nMSE, sign
agreement, Pearson and Spearman value for their arms. A negative result rests on
its checkpoints as much as a positive one does. They are retained with their
hashes, configs and metrics, and released with their diagnostic status stated.

## 8. "Sampled Eq. (24)" was the wrong name for it

**Was:** the sampled construction was repeatedly called "the sampled Eq. (24)
target".

**Is:** Eq. (24) (`eq:finite-target-logratio`) is the **deterministic** finite-pool
target

    h*_t(x,i,i') = log(p*_x(i)/p_{t,x}(i)) - log(p*_x(i')/p_{t,x}(i'))

The Bernoulli-and-opponent construction the pair builder ships is an *estimator*
of that quantity, not the quantity itself, and calling it "Eq. (24)" attributed
a sampling scheme to an equation that has none. Verified against the compiled
labels: (22) `finite-pool-value`, (23) `direct-inner`, (24)
`finite-target-logratio`, (25) `log-ratio-change`, (26) `regression-loss`,
(27) `finite-target-identity`, (29) `dual-update`.

The `canonical` target mode added in this round computes Eq. (24) exactly --
matching the solver's own optimizer output to 3.6e-15 -- so the manuscript's
equation and the code now denote the same object under the same name.
