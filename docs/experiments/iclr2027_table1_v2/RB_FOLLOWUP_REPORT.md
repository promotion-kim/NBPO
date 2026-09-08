# NBPO RB follow-up: what was measured, what was fixed, what is still open

Covers the work requested in `NBPO_RB_followup_prompt.md`. The generalization
curve is still running; every other section is final. Sections map to the eight
items the request asked the completion report to contain.

## 1. Gate table

Requirement, all four simultaneously on the test half after selecting on
validation: `normalized_mse_var < 0.90`, `sign_agreement > 0.65`, Pearson `> 0`,
Spearman `> 0`. Solver-side items are separate and all pass.

| arm | target | val nMSE | test nMSE | test sign | test Pearson | test Spearman | worst surplus | gate |
|---|---|---|---|---|---|---|---|---|
| `pi_t` (start) | — | — | — | — | — | — | −0.00350 | — |
| eta = 0.1 | sampled | 2.319 | 2.362 | 0.485 | −0.011 | −0.018 | −0.02052 | fail |
| eta = 0.3 | sampled | 1.121 | 1.182 | 0.490 | −0.030 | −0.008 | −0.02061 | fail |
| eta = 1.0 | sampled | 1.020 | 1.026 | 0.505 | −0.026 | +0.001 | −0.01972 | fail |
| lr = 2e-7 | sampled | 1.006 | 1.016 | 0.497 | −0.008 | −0.004 | −0.02080 | fail |
| lr = 1e-6 | sampled | 1.020 | 1.028 | 0.512 | +0.003 | +0.026 | −0.02060 | fail |
| lr = 1e-5 † | sampled | 22.308 | 18.623 | 0.517 | −0.008 | +0.020 | −0.03013 | fail |
| RB, 300 † | RB | 1.070 | 1.126 | 0.535 | +0.025 | +0.055 | −0.02002 | fail |
| RB, clip 100 † | RB | 1.105 | 1.093 | 0.542 | +0.064 | +0.089 | −0.02016 | fail |
| RB, 1200 † | RB | 1.074 | 1.077 | 0.560 | +0.089 | +0.114 | −0.01847 | fail |

† diagnostic, outside the pre-declared grid, never a selection candidate.

Validation correlations are reported above and were what selection used; the
positive test correlations of the daggered arms are **not** offered as evidence
of a validation pass. Solver side, on the SafeRLHF smoke pool: projected KKT
3.55e-15, extra-map residual 3.25e-13, target identity 1.78e-15, `inner_solver =
exact`, `dual_solver = root`, eta applied once, sequence-sum reduction, artifact
provenance bound and verified.

Algorithm 1's own acceptance rule -- held-out surplus positive for every
objective -- is checked separately and fails for every arm, all of which sit
below `pi_t`.

## 2. Full MSE decomposition

Both exact identities, computed for every arm, closing to ~1e-14:

```
MSE = E[T^2] + E[h^2] - 2 E[hT]
MSE = Var(T) + Var(h) - 2 Cov(h,T) + (mean(h) - mean(T))^2
```

The earlier write-up used `Var(T) + E[h^2] - 2 Cov`, exact only at `mean(T) = 0`.
Measured `mean(T)` is +0.040 (validation) and +0.089 (test) at eta = 1, and that
formula is off by up to 1.8e-2. `nMSE_zero` (dividing by `E[T^2]`) is reported
alongside `nMSE_var`; they differ in the fourth decimal here, so nothing turns on
the choice. Each arm is scored at its OWN training eta -- arms at different eta
solved different proximal problems and must not share one target.

## 2b. How much of the correlation survives uncertainty

Pairs within a prompt share its responses and its target row, so they are not
independent draws; the resampling unit is the prompt. Bootstrap Pearson, 2000
resamples over the 150 held-out test prompts:

| arm | test Pearson | 95% CI | excludes 0 |
|---|---|---|---|
| eta = 1.0, sampled | −0.026 | [−0.056, +0.006] | no |
| lr = 2e-7, sampled | −0.008 | [−0.041, +0.027] | no |
| RB, 300 | +0.025 | [−0.031, +0.084] | **no** |
| RB, clip 100 | +0.064 | [+0.007, +0.123] | barely |
| RB, 1200 | +0.089 | [+0.039, +0.144] | **yes** |

This corrects a reading in the earlier note. The 300-step Rao-Blackwell arm was
described as the first with positive correlation on both halves; its interval
includes zero, so it is not distinguishable from no correlation at all. Only the
1200-step arm is. That is what makes the under-training reading a measurement
rather than an impression -- and it is also a reminder of the size involved: a
correlation of 0.089 is a long way from a normalized MSE below 0.90.

This is a reporting interval, not a new gate.

## 3. Target identity and noise audit

The three quantities the manuscript's Eq. (24)-(27) relate, checked against the
artifacts rather than assumed:

| check | result |
|---|---|
| `m` (optimizer log-ratio) vs `mQ` (KKT form) at `nu_update` | max residual 5.3e-15 |
| same, at `nu_final_policy` | max residual 3.9e-12 |
| canonical target is realizable by some policy log-ratio | max residual 8.9e-15 |
| canonical pairs vs `m` | max difference 3.6e-15, Pearson 1.0000000000 |

Randomness budget per pair: `Var(m)` 6.427, Bernoulli variance 38.741,
comparator variance 0.109. So:

| estimator | integrates | `corr(T,m)^2` | conditional-variance ratio |
|---|---|---|---|
| `sampled` (Eq. 24) | nothing | 0.1425 | 0.1420 |
| `rao_blackwell` | the Bernoulli flips only | 0.9840 | 0.9833 |
| `canonical` (added here) | flips and comparator | 1.0000 | 1.0000 |

**The shipped `rao_blackwell` mode is a partial Rao-Blackwellization**: it still
draws one comparator per (row, objective). The residual 1.6% is that draw, not
"inevitable neural noise", and it is removable in closed form from the payoff
tensor already on disk -- which is what `--target-mode canonical` now does.

## 4b. The surplus code checks out against the solver

The held-out surplus numbers are a headline result, so the implementation is
validated against a value the solver already reported rather than trusted. Fed
the solver's own optimizer output `pi_star`, it reproduces both quantities
exactly:

| quantity | solver's record | recomputed here | difference |
|---|---|---|---|
| surplus, helpfulness | 0.14354640 | 0.14354640 | 0 |
| surplus, harmlessness | 0.14819243 | 0.14819243 | 0 |
| disagreement `d`, helpfulness | −0.03228067 | −0.03228067 | 0 |
| disagreement `d`, harmlessness | −0.03386381 | −0.03386381 | 0 |

Same soft-min game value, same disagreement construction, same pool weighting.

## 5. Actual optimizer update comparison

`clip100` changed `max_grad_norm` from 1.0 to 100 with everything else fixed and
moved held-out nMSE from 1.126 to 1.093 and test Pearson from +0.025 to +0.064.
Pre-clip gradient norms are ~9.4e3 in both. Global-norm clipping preserves
direction, and the realized update also depends on the learning rate and
Adafactor's internal scaling, so this is an interaction, not a cause. The earlier
claim that clipping "makes every update a fixed-size normalized step" is
withdrawn (see `INTERPRETATION_CORRECTIONS.md`).

## 6. Causes established, and causes still open

**Established.**
- The trainer can fit this target: 8 prompts, 224 pairs, 200 updates gives an 87%
  loss reduction against its own `h = 0` baseline, with `h` RMS rising to match
  the target scale. Optimizer, autograd, loss and precision are not blocking.
- Longer training helps monotonically: 300 -> 1200 updates moves test Pearson
  +0.025 -> +0.089 and Spearman +0.055 -> +0.114, with train and validation
  improving together.
- Estimator noise is not the blocker on its own: a 98%-predictable target still
  fails at 300 updates.
- Clipping is not the cause: 100x relaxation barely moves the result.
- dtype is not the cause: the base checkpoint already declares bf16, both paths
  load bf16, and an explicit setting reproduces the run to six decimals.

**A real defect, found and FIXED.** The zero-step identity failed: at
`learning_rate = 0` the weights never change, so `h` must be exactly 0, and it
measured RMS 0.30-0.61. The cause is now established (see
`ZERO_STEP_ROOT_CAUSE.md`): `log pi(a)` came from the online forward and
`log pi_t(a)` from the precompute cache, whose batches were grouped differently,
and in bf16 a response's log-probability depends on its batch neighbours. Two
terms, measured apart -- bf16 quantization of the accumulated sum (long-response
log-probabilities came back as exact integers, differences exactly one ulp), and
the bf16 forward's own dependence on batch shape.

Fixed by accumulating the log-probability in float32 and, decisively, by
forwarding a frozen copy of `pi_t` through the SAME collated batch as the policy.
Zero-step now gives `h` exactly 0.000000 at every step, and the new
`nbpo/ref_online_minus_cache_rms` diagnostic reproduces the old path's `h` to six
decimals -- so the entire zero-step error was this and nothing else.

Whether the fix changes the gate is a separate question, measured by the paired
before/after runs and not assumed. Note also that the gate drew both of its terms
from the same cached path, which made it internally consistent but was never
independently verified, and a training signal that was wrong still determined
where the earlier policies ended up.

**Open.** Whether the residual failure is prompt-level generalization or
remaining budget. The equal-compute prompt-count sweep now running is the
measurement; the arithmetic that motivates it is that the tiny fit saw each pair
14 times and the 700-prompt run saw each pair less than once.

## 7. Protocol v2: not adopted

Nothing has been frozen as protocol v2. The diagnostics that improved things
(canonical target, longer training) are recorded as candidates, and adopting them
would require freezing the change, its rationale, the selection data and the
budget first, then running a confirmation. No result here is described as if it
had been pre-declared.

## 8. Artifacts

- Preference models: <https://huggingface.co/promotion/nbpo-saferlhf-preference-models>
  revision `33a834c38e47591724345a07d857953d7c1eb9ae`, public, all six reproduce
  their local logits exactly.
- Policy arms (diagnostic): <https://huggingface.co/promotion/nbpo-policy-diagnostics>
  revision `2a6b832280f2e6413d444b296a17eaf1642c0c6a`, public, nine checkpoints
  verified byte-identical anonymously.
- Manuscript: `nbpo_iclr/main_v6.tex`, 22 pages, body ends on page 9, 0 undefined
  references, 0 overfull boxes.

## Resources

MLXP `p-aipr`, pod `nbpo-judge` on node `h200-03-w-aa21`, **4x H200**. Getting
the fourth required deleting two of the user's own pods (`polyedit-grpo`, idle
for four days, and `nbpo-judge2`, which grabbed a freed GPU the moment it
appeared). Both H200 nodes in the zone run at 8/8 allocated, which is why a
1-GPU request had been pending for 29 hours. Storage: `sjkim` at `/work` and
`sjkim-data` at `/data`; **`sjkim-workspace` does not exist** -- the two above are
the only PVCs under that name.
