# P0: independent solver and aggregation-target audit (UF-4)

Recomputed from the stored arrays in NumPy under the manuscript's own definitions,
with no call into the solver's code, by
`scripts/experiments/uf4_20260910/audit_uf4_solver.py`.

```
r_k(x,j)    = sum_i pi(x,i) A_k(x,i,j)                                Eq. (6)
V_k(pi)     = mean_x [ -beta_k logsumexp_j( log mu(x,j) - r/beta_k ) ] Eq. (8)
d_k         = V_k(mu) from the reference-as-learner tensor             Eq. (10)
s_k(pi)     = V_k(pi) - d_k
F(pi)       = sum_k log s_k(pi)
D(pi||pi_t) = mean_x KL(pi_x || pi_t,x)
J(pi)       = F(pi) - D(pi||pi_t)/eta                                  Eq. (14)
```

Eq. (14) is what the stage maximizes: `max sum_k log u_k - D/eta` s.t. `u_k <= s_k(pi)`,
so `u_k = s_k(pi)` at the optimum. K = 4, pool 8, beta = 0.25 on every objective,
eta = 1, mu uniform, reference construction `shared_pool`.

## The question this audit was written to answer

From the rounded surpluses alone, the L1-matched utilitarian solution has about
+0.024 higher *unregularized* Nash welfare F than the claimed Nash optimizer. If
that survived the proximal penalty, the Nash solve would not be solving its own
objective and every dependent experiment would be built on it.

It does not survive. Measured, full precision:

| Split | Arm | F | D | J | min s_k |
|---|---|---:|---:|---:|---:|
| train (10,000 prompts) | Nash | -8.493723006635 | 1.080071681041913 | **-9.573794687676912** | 0.09055809429288733 |
| train | L1-matched utilitarian | -8.469298363880046 | 1.1064233613605443 | -9.57572172524059 | 0.09046996022767029 |
| dev (1,000 prompts) | Nash | -8.467210287248182 | 1.0877070924975754 | **-9.554917379745758** | 0.09116149310786731 |
| dev | L1-matched utilitarian | -8.44212288367166 | 1.1149326299112594 | -9.557055513582918 | 0.09107785623243089 |

Gaps, utilitarian minus Nash:

| Split | F | D | J | verdict |
|---|---:|---:|---:|---|
| train | +0.024424642754953 | +0.026351680318631 | **-0.001927037563679** | Nash attains the larger J |
| dev | +0.025087403576523 | +0.027225537413684 | **-0.002138133837160** | Nash attains the larger J |

The utilitarian candidate does buy higher raw F, and the +0.0244 figure reproduces
the value predicted from the rounded surpluses. It pays for it with +0.0264 more
proximal KL, and at eta = 1 that enters J at full weight, so the ordering reverses.
No solver defect. The margin is small (0.0019 on train) and is reported as such.

## Consequence for the "matched" control

**L1-matched coefficients do not match realized movement.** The utilitarian arm
travels 2.44% further from the reference in mean prompt KL (1.1064 vs 1.0801 on
train). Any claim that the two arms received the same budget must say *coefficient
L1 matched* and report realized D separately, as here.

## Certificates, recomputed independently

| Quantity | train | dev |
|---|---:|---:|
| Nash dual residual, max_k \|s_k(pi(lambda)) - 1/lambda_k\| | **3.735900477863652e-14** | 9.82533525989801e-04 |
| pi rebuilt from lambda vs stored pi, L-inf | 3.19e-14 (dev), see JSON | 3.189202374409561e-14 |
| Nash inner stationarity spread, max over prompts | 1.8801482593033825e-10 | 9.622569407952142e-11 |
| Utilitarian inner stationarity spread, max | see JSON | 9.965539504719345e-11 |
| Simplex deviation, max | ~2e-16 | 2.220446049250313e-16 |
| Minimum probability | > 0 | 2.592352759358463e-10 |
| Target identity `g = log(pi/pi_t)`, max abs | 0.0 | 0.0 |

The dual certificate is built by rebuilding `pi` from `lambda` alone through the
exponential map, iterated to its own fixed point (23 iterations, final map delta
4.0e-15), and only then comparing `s_k(pi(lambda))` with `1/lambda_k`. The stored
policy is never consulted, so this is a real optimality condition and not the
identity that defines lambda. That rebuilt policy reproduces the stored one to
3.2e-14, which is what certifies the inner solve.

**What the solver's reported 1.39e-16 measures:** it is the solver's own internal
residual on the train split. The independent recomputation on train gives
3.7e-14, the same conclusion at a different scale. On dev the same expression is
9.8e-4, and that number is **not** a solver error: lambda is fitted on train and
held fixed on dev by design, so `s_k = 1/lambda_k` is not expected to hold there.
The dev value measures the train-to-dev surplus shift. The dev split still gets
full stationarity and feasibility checks, which it passes.

Antisymmetry of `A_ref` (residual 0.0) and its zero diagonal are imposed by
construction. They are invariants, not evidence that the preferences or the
optimizer are correct.

## Does aggregation change the target?

| Comparison | mean TV | median | p95 | max | sym. KL mean | pair-target diff RMS / target RMS |
|---|---:|---:|---:|---:|---:|---:|
| Nash vs L1-matched utilitarian, train | 0.019505 | see JSON | see JSON | see JSON | see JSON | **3.437%** |
| Nash vs L1-matched utilitarian, dev | 0.019872 | | | | | 3.407% |

For scale, the same contrast on the SafeRLHF two-objective panel was mean TV
0.0008 and 0.3% of target RMS. UF-4 separates the two rules about 24x more in TV
and about 11x more in relative pair-target RMS. Its Nash coefficients are also
genuinely unequal: lambda = (7.4627, 11.0426, 8.1138, 7.3043), L1 = 33.9235, with
truthfulness the binding objective at surplus 0.0906, against SafeRLHF's nearly
equal (6.325, 6.186).

That is a target-level statement only. No TV threshold is declared here, and none
should be invented after seeing these numbers. The quantity it has to be read
against is the neural realization error: at update 200 of the running Nash arm the
fitted log-ratio magnitude is 2.251 against a target magnitude of 2.200 with
residual magnitude 1.901, so the projection currently misses its target by an
amount comparable to the target itself -- two orders of magnitude larger than the
3.4% separation between the two aggregation targets. On this evidence a
policy-level Nash-versus-utilitarian difference is unlikely to be resolvable at
the present fit quality, which is a reason to keep the contrast as a control
rather than to spend three seeds on it.

## Full precision

`solver_audit_train.json`, `solver_audit_dev.json`.
