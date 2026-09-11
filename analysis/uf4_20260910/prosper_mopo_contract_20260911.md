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
values at the same beta**, not a new algorithm. Planned implementation reuses
the verified machinery at X=1 per prompt rather than writing a second solver:
11,000 independent tiny solves (10k policy_train + 1k policy_dev), each fitting
its own weights. The existing global solve does 10k per-prompt inner solves in
about 6 minutes on 32 workers, so a per-prompt outer fit is expected in the tens
of minutes on CPU, running beside training.

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
