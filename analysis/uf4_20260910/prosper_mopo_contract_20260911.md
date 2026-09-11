# PROSPER and MOPO: what a faithful adaptation requires

Written 2026-09-11 22:15 KST. These two rows are the only remaining blocker on
"first complete main-body evaluation", so this records what each contract
actually is before any code is written. Nothing here is measured yet.

## PROSPER

The manuscript states the criterion (app:uf-protocol):

    min over w: X -> Delta_K, and over nu, of
        E_x [ sum_k w_k(x) J_{k,x}(pi, nu) + beta * KL(nu(.|x) || pi_ref(.|x)) ]

Two things follow, and the second is what makes this tractable.

**The inner weight minimisation collapses to a per-prompt minimum.** A linear
function on the simplex attains its minimum at a vertex, so
`min_{w in Delta_K} sum_k w_k J_{k,x} = min_k J_{k,x}`. PROSPER's adversarial
weights are therefore the per-prompt worst objective, which is exactly the
distinction the manuscript's two-prompt example draws: surplus vectors (.2, 0)
and (0, .2) average to (.1, .1) and win under global Nash, while their average
prompt-wise minimum is 0 against .08 for the policy that is (.08, .08) on both.

**The problem decomposes across prompts.** NBPO's dual lambda is shared across
prompts, which couples them into one joint solve. PROSPER has no shared weight,
so each prompt is an independent maxmin over that prompt's 8-candidate simplex
against the KL-regularised opponent. That is the same per-prompt inner problem
the current solver already computes; what changes is that no global dual is fit
on top of it.

So a faithful adaptation is **per-prompt absolute_maxmin on the adaptive-game
values at the same beta**, not a new algorithm.

### Correction, 22:45 KST: this is not a flag on the existing solver

The earlier note said the per-prompt solve could reuse the existing pipeline at
X=1. The solve itself can. The *artifact* pipeline cannot, and the obstacle is
structural rather than cosmetic:

    FinitePoolSolution.weights: torch.Tensor   # (K,)

is one global weight vector, and `write_generic_solution_artifact` records it as
`"lambda_raw": [float(v) for v in res.weights]`. `load_canonical_artifact`
validates against that record and `build_rows` is handed `result.weights`
as (K,). Per-prompt weights are (K, X), so four places on the write path would
have to change, all of them shared with the arms already published.

What rescues the design is that **the training target does not depend on the
objective weights at all**. The canonical target is

    log(w_a / c_a) - log(w_b / c_b)

over the solved candidate masses against the uniform 1/8 centre. Checked against
a released row: recorded target -0.150054122, recomputed from the masses
-0.150054122, exact. The weights determine *which* policy is solved; they do not
enter the target once it is solved.

So the implementation is a separate `solve_prosper_targets.py` that:

  * calls the verified `solve_finite_pool` once per prompt on a one-prompt
    representation, with `absolute_maxmin` and the L1 norm matched to the Nash
    dual, exactly as the utilitarian and global-maxmin controls are matched;
  * assembles the canonical target itself from the per-prompt solved masses,
    which is the same closed form the released rows satisfy;
  * writes per-prompt weights as a (K, X) array in its own provenance file
    rather than forcing them through a field shaped for one vector;
  * leaves `solve_uf4_targets.py` and every published artifact untouched.

Before it is trusted it has to reproduce a global-weight case: run the same code
path with one shared weight vector and check the targets match
`targets/nash_v1/pairs` row for row. Reimplementing the target formula is the
one place this design can silently diverge from the arms it is compared with, so
that check is not optional.

One honest consequence of no shared dual: policy_dev is no longer a held-out
measurement of a fit made on policy_train, because each prompt fits its own
weights on whichever split it is in. The dev split still measures generalisation
of the *neural* fit, but not of a dual. That difference belongs in the row's
description.

It must NOT be labelled as replacing global game-maxmin. Global maxmin changes
the compromise over shared prompt-averaged values; PROSPER also changes the
order of aggregation across prompts. Both rows are needed to separate those.

Open item: whether beta for PROSPER is the UF beta (0.25) used by the
adaptive-game arms. Keeping it matched is the defensible default, since the
contrast being drawn is aggregation order, not opponent temperature.

## MOPO

Different in kind, and not expressible by choosing an aggregation. Per the
manuscript, MOPO uses pairwise preferences with no point-wise reward, selects a
**primary objective** with **lower-bound constraints** on the others, and uses a
**lagged reference** refreshed on a schedule (June 2026 revision).

That is a constrained problem, not a bargaining one: maximise the primary
objective subject to V_k >= d_k + eps_k for the others, against a reference that
lags rather than stays frozen. Three declarations are required before it can be
implemented, and none of them can be read off our artifacts:

1. which objective is primary. It must be chosen without touching final
   results; the defensible route is the declared policy_dev split.
2. the lower bounds eps_k, and whether they are absolute or reference-relative.
3. the lagged-reference refresh schedule, which interacts with our single outer
   stage -- a lag has no meaning at T=1 unless the lag is within-run.

Until those are declared, any MOPO row would be an implementation we invented
rather than the published contract, so the row stays unmeasured and the
manuscript continues to say the adaptation is required rather than done.

## Consequence for the schedule

PROSPER is implementable now, from artifacts already on disk, with no new
feedback and no new GPU cost beyond the training run itself. MOPO is blocked on
three declarations. Reporting them separately is more honest than reporting one
joint "PROSPER/MOPO pending".
