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

## 9. "Only the 1200-step arm is significant" was wrong

**Was:** of the bootstrap intervals, only `RB, 1200` was said to exclude zero.

**Is:** by the table's own numbers, two do -- `clip100` at [+0.007, +0.123] and
`RB 1200` at [+0.039, +0.144]. `RB 300` at [−0.031, +0.084] does not. The claim
misread the table it was standing on.

Two further things that sentence conflated:

- These are **nominal** intervals for individual correlations, computed after
  looking at several arms. Nothing here corrects for having explored nine arms,
  so "excludes zero" is a description of one interval, not a multiplicity-adjusted
  finding.
- An interval on arm A excluding zero and an interval on arm B including zero
  does **not** establish that A and B differ. Testing a paired difference is a
  different computation, on the paired quantity, and it was not done.

## 10. A significant correlation is not a passing gate

Pearson +0.089 is weak alignment. Even the best affine recalibration of `h` on
the same sample leaves a variance-normalized MSE of at least
`1 − r² = 1 − 0.089² ≈ 0.9921`, against a gate of 0.90. So no amount of
significance on that correlation implies the gate is reachable at that alignment,
and it says nothing about whether the policy is useful.

This is arithmetic for interpreting the number, not an instruction to fit a
calibration on the test half.

## 11. Regression and frozen-pool surplus are complementary, not independent

**Was:** "two independent criteria give the same answer."

**Is:** both are computed on the same held-out prompts, the same frozen response
pool and the same frozen preference models. They answer different questions --
does `h` track the target, and does the induced pool distribution raise the game
values -- but they share nearly all of their inputs. Agreement between them is
weaker evidence than the phrase "two independent criteria" suggests, and neither
is a measurement of generation quality: nothing here decodes a new response or
consults an independent judge.

## 12. The surplus table was partial, and "both objectives worse" needs the values

`min_k s_k < 0` means at least one objective fails the acceptance condition. The
per-objective numbers are what license a statement about both, and the surplus
table covered four arms while the regression table covered nine. Claims of rank
agreement across arms need the same arms in both tables; `clip100`'s surplus and
the full regression columns were missing when that comparison was drawn.

## 13. Stop saying dtype, clipping and the estimator are "rejected"

Each was a single change in one setting, and that is all any of them showed:

| change | setting | result |
|---|---|---|
| `max_grad_norm` 1.0 -> 100 | RB target, 300 updates | test nMSE 1.126 -> 1.093 |
| target estimator sampled -> RB | eta 1, 300 updates | test nMSE 1.026 -> 1.126 |
| explicit `torch_dtype: bfloat16` | zero-step probe | identical to six decimals (already bf16) |

"Rejected as the cause" overstates all three. What the zero-step work later
established is separate and specific: the bf16 **accumulation** of the sequence
sum, and the bf16 forward's dependence on batch shape, together account for the
zero-step error exactly.

## 14. "Evaluation unaffected" was asserted, not verified

**Was:** "this contaminates the training signal only; the gate computes both
terms through the cache, so evaluation is unaffected."

**Is:** the gate's two terms being drawn from the same cached path makes it
internally consistent, which is not the same as correct, and it was never
separately verified. Two things follow. The evaluation path needs its own check.
And a training signal that was wrong can still have determined where the final
policies ended up -- consistency of the ruler does not undo a mis-measured build.

## 15. The surplus evaluator used the wrong pool mapping, and the result reversed

**Was:** "every arm sits below `pi_t`; training moves the held-out game values away
from where they started; Algorithm 1 rejects all of them."

**Is:** that was produced with `p_theta = softmax(log pi_theta)` over the pool.
The finite-pool problem's induced distribution is the RELATIVE update

    p_theta(i)  proportional to  p_t(i) * exp( log pi_theta(i) - log pi_t(i) )

and the two coincide only when `p_t` is the conditional LM mass over the pool.
Here `p_t` is uniform -- every row of `pi_t.npz` is 0.125 -- so the old mapping
scored the LM's absolute log-probabilities, which track response length and
content, instead of what training changed. It also failed the obvious check:
at `theta = theta_t` it did not return `p_t`.

The corrected evaluator runs two unit checks before reporting anything and
refuses to continue if either fails:

| check | error |
|---|---|
| `h = 0` recovers `p_t` | 0 |
| `h = log(p*/p_t)` recovers `p*` | 4.4e-16 |

With the mapping fixed, held-out worst-objective surplus:

| policy | validation | test | accepts on test |
|---|---|---|---|
| `pi_t` | −0.01264 | −0.00350 | no |
| `pi_star` (solver optimum) | +0.13926 | +0.14486 | **yes** |
| RB, 1200 | −0.00808 | **+0.00160** | **yes** |
| canonical, N=700, 1200 | −0.00995 | **+0.00230** | **yes** |
| RB, clip 100 | −0.01301 | −0.00007 | no |
| eta = 1.0, sampled | −0.01217 | −0.00572 | no |

Three things follow, and none of them is "the arms pass".

- **No arm accepts on validation.** Validation is what selects; a test-half pass
  is not a validation pass and is not treated as one.
- The paired improvements over `pi_t` are small and mostly not distinguishable
  from zero: on test, `canonN700` gives +0.00580 with a prompt-clustered interval
  of [−0.00001, +0.01111].
- `pi_star` reaches +0.145 on held-out prompts, so the target is feasible out of
  sample. Whatever fails, it is not the oracle's held-out feasibility.

`pi_t` scoring slightly negative everywhere is expected rather than a defect: the
disagreement point is estimated from an independently constructed
reference-reference tensor, not from `pi_t` itself.
