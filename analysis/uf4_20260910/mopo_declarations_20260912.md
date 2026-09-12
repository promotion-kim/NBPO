# MOPO declarations, taken from the paper

Source: Agnihotri, Jain, Ramachandran, Wen, *Multi-Objective Preference
Optimization: Improving Human Alignment of Generative Models*, arXiv 2505.10892
(ICML 2026); cited in the manuscript as `agnihotri2026mopo`. Read 2026-09-12.
Every value below is the paper's, not ours; the two places we depart are marked
**DEPARTURE** with the reason.

## The algorithm, as published

Constrained optimization (their Problem 1), with `p` the primary objective's
pairwise preference and `q ∈ [0,1]^{K-1}` the secondary ones:

```
max_π   E_{x~ν, y~π(·|x), y'~μ(·|x)} [ p(y ≻ y' | x) ] − τ · KL(π ‖ π_ref)
s.t.    E_{x~ν, y~π(·|x), y'~μ(·|x)} [ q(y ≻ y' | x) ] ≥ b
```

Their Algorithm 1, verbatim:

```
Input: Dataset D, batch size M, learning rate η, epochs T, lag t0, relaxation parameter β.
for t = 1..T:
    χ(t) = [χ(t-1) − η ∇_χ J(χ; ρ_λ(t-1))]+          (Eq. 8)
    λ(t) = [λ(t-1) − η ∇_λ J(λ; χ(t))]+              (Eq. 9)
    compute ρ_λ(t)(·)                                 (Eq. 3)
    ψ(t) = ψ(t-1) − η ∇_ψ J_{ρ_λ(t)}(π_ψ)            (Eq. 9)
    if t mod t0 == 0:
        b ← β^T G(ρ^(t−t0))
        π_ref ← π_ψ^(t−t0)
```

with the optimal importance ratio (their Eq. 3)

```
ρ*_λ(y) = exp( τ^{-1} [ E_{y'~μ}[ p(y ≻ y') ] + λ^T E_{y'~μ}[ q(y ≻ y') ] ] − 1 )
```

and the policy step (their Eq. 7) an importance-weighted behaviour cloning,
`max_π E_{y~π_ref}[ ρ*(y) log π(y) ]`.

Their Table 3 hyperparameters for the text-generation experiments:

| quantity | paper's value |
|---|---|
| Finetuning steps | 10000 |
| Batch size | 8 |
| Regularization τ | 0.08 |
| Constraint lower-bound ball ε | 0.15 |
| Constraint relaxation β | 0.9995 |
| Reference policy lag t0 | 500 |

Ablation grids (their Figures 8 and 14): β ∈ {0.9, 0.95, 0.99, 0.999} and
t0 ∈ {1, 100, 1000, **∞**}; the paper reports MOPO stable across all of them.

## Declaration 1 — which objective is primary

**Instruction following.**

The paper fixes the primary objective *positionally*, not semantically: it
writes `p_K ≡ p` for "the K-th (the primary) objective" and `q` for the
remaining `K−1`, and in the experiments states "we let the primary objective be
`r₁` and constrain preferences w.r.t. `r₂`" — its result columns are named
`(r₁, r₂)` = (helpful, harmless), (humour, harmless), (no_hallucinate,
faithful), (pref1, faithful), i.e. `r₁` is the first objective in each dataset's
own declared order.

Our objective order was fixed before any of these results existed and is used
everywhere else in the campaign — instruction following, truthfulness, honesty,
helpfulness — so applying the paper's rule mechanically makes **instruction
following** primary and constrains the other three.

This is the point of the declaration: the primary objective is chosen by the
paper's positional rule applied to an order frozen in advance, **not** by
looking at which objective would flatter MOPO. Picking the weakest objective
(honesty, which binds `min_k W_k` for all four of our mechanisms) would have
been choosing a baseline's hyperparameter from our own reported metric, which is
the same move already disqualified for the cross-play competitor.

## Declaration 2 — the lower bounds

**Neither absolute constants nor bounds relative to the frozen base: the
paper's own adaptive self-referential schedule.**

Their adaptive constraint schedule sets, every `t0` steps,
`b = β^T G(ρ^(t−t0))` — the bound on each secondary objective is `β` times *the
value the policy itself achieved `t0` steps ago*, with β = 0.9995. Secondary
values are expected preference probabilities against the behaviour policy μ
(which for us is the frozen candidate pool's generator), so they live in [0,1]
with 0.5 as parity. The constraint lower bound also carries the KL ball
ε = 0.15 of their Problem (5), the radius within which the adversarial
underestimating distribution `π_k` may move from `π_ref`.

We take β = 0.9995 and ε = 0.15 and τ = 0.08 verbatim.

## Declaration 3 — the lagged-reference schedule

**Main-body row: t0 = ∞ (no refresh), which is one of the four settings the
paper itself ablates. A lagged variant at the paper's ratio is queued
separately.**

The paper's t0 = 500 is *within-run* over 10000 finetuning steps — twenty
refreshes — and refreshing needs no new data because expectations are reweighted
by `ρ_lag,ref = π_ψ^(t−t0)/π_ref`. That resolves the objection recorded earlier
in this campaign that a lag is meaningless at one outer stage: it is meaningless
only for a *stage*-level lag, not for a step-level one.

**DEPARTURE.** Our matched recipe is 1250 contiguous updates on 10000 policy
prompts, and our pipeline solves targets offline and then trains, so a
step-level refresh would require re-solving the dual mid-run against a policy
snapshot. Rather than silently change the recipe, the main-body row uses
**t0 = ∞** — the paper's own no-lag setting, which its Figure 8/14 ablations
report as stable — with the dual solved to convergence against the shared frozen
base reference. A second row at **t0 = 250** (five refreshes across 1250
updates, inside the paper's tested {100, 1000} bracket), implemented through the
existing outer-stage runner, is queued as a declared variant so the paper's
default mechanism is attempted rather than dismissed. Both rows are labelled.

**DEPARTURE.** Steps and batch size are ours, not the paper's: 1250 updates at
the campaign's global batch instead of 10000 at batch 8, and full-parameter
tuning instead of LoRA. This is the same matched-budget contract every other arm
in the table runs under, and changing it for one baseline is what would make the
comparison unfair.

## What was already implemented

`mnpo_trainer.py` already carries a `loss_type == "mopo"` branch citing Eq. (7)
and Eq. (3): it expects a column holding `ρ(y)` and applies
`losses = −ρ(y) · log π(y)`, the importance-weighted behaviour cloning. The
missing piece is the target solver that produces `ρ(y)` from our teacher tensor
under the dual above — the MOPO analogue of `solve_prosper_targets.py`.

## Open implementation note (not a declaration)

The trainer's MOPO loss is `losses = −ρ(y) · log π(y)` on the *chosen* side
only; the rejected side enters no term. Our shared pair enumeration is
`itertools.combinations(range(8), 2)` with `chosen = i < j`, under which
candidate 0 appears as `chosen` seven times and candidate 7 never. Feeding
MOPO the shared pair rows would therefore weight the behaviour-cloning estimate
by a candidate's index rather than by its ρ, which is not MOPO.

So MOPO needs its own row construction: one row per (prompt, candidate), 10000 ×
8 = 80000 rows, each carrying that candidate's ρ(y). This must be written and
checked before the arm is queued — the same "read the callee before queueing a
batch" rule that caught the DPO, HarmBench and spaCy failures. It is the reason
the arm is not queued in the same turn as these declarations.

## What running it showed (2026-09-12, solver verified)

The dual was solved on both splits. Three things came out of it, and two are
corrections to what is written above.

### The lower bound is exact, after three of my own errors

The closed form `max_χ [−χ log E exp(−ρq/χ) − χε]` now agrees with a direct
constrained minimisation to **5.7e-14** relative, swept over KL radii
{0.15, 0.01, 1e-3, 1e-5}. Getting there took three fixes, all caught by the
check rather than by reading the code: the check's own sign (Problem (5)
*minimises*, so the direct value is `+Σ w v`; writing `−Σ w v` gave a relative
residual of exactly 2), a log-spaced grid for the χ maximisation (replaced by
ternary search on log χ, since the objective is concave), and a χ bracket capped
at 1e2 when the optimal χ grows like 1/√ε (a 1e-5 radius needs χ ≈ 200).

### CORRECTION 1 — ε = 0.15 cannot be transplanted, and neither can ε > 0

Measured on our own tensors, the paper's ε = 0.15 costs **0.0612 / 0.0742 /
0.0831** of the three secondary values at the reference policy, while the
paper's β = 0.9995 allows a slack of **2.5e-4** — a factor of 240 to 330. The
constraint is violated before optimisation starts, λ runs away linearly (73.8
after 300 iterations and still rising) and ρ collapses to one-hot, which turns
MOPO's policy step into best-of-8 cloning.

Calibrating ε so the robustification consumes exactly β's slack at the reference
(ε = 1.48e-6) fixes stage 0. It does **not** survive the ratchet: once ρ spreads
out, the adversary has more room, the gap grows past (1−β)·G again, and the
solver's feasibility guard fires at stage 1 with a violation of 5.4e-4.

So the robustification is dropped: **ε = 0**, the paper's Problem (3) constraint
on the estimated values directly. The justification is the paper's own — it
introduces the χ step to make the policy "robust against these estimation
errors" in `q̂` collected from a finite dataset, and our `q̄` is an exact
deterministic teacher evaluation over the complete 8×8 block of every prompt,
not a bootstrap estimate from sampled comparisons. The departure is recorded
with the number that forces it rather than asserted.

### CORRECTION 2 — t0 is inert here, so the main row's t0 = ∞ needs no apology

Simulating the ratchet `b ← β·G(ρ_prev)` over six stages: `b` climbs from
[0.4998, 0.5000, 0.5002] to [0.5699, 0.5919, 0.6033] and **λ stays exactly 0 at
every stage**, with ρ and the primary value unchanged to five decimals. The
fixed point is reached in one step. So t0 = ∞ and t0 = 250 produce the identical
ρ, and the earlier worry that t0 = ∞ makes the main row a degenerate limit does
not apply — measured, over six ratchet stages, rather than argued.

### The finding: MOPO's constraints never bind on UF-4

Why λ = 0 is not a solver artifact. Within each prompt, across the eight
candidates, the teacher's primary and secondary values are nearly rank-identical:

| secondary | within-prompt Spearman vs IF | prompts positive | prompts below −0.5 |
|---|---|---|---|
| truthfulness | 0.870 | 99.8 % / 99.6 % | 0.02 % / 0 % |
| honesty | 0.912 | 100 % / 99.9 % | 0 % / 0 % |
| helpfulness | 0.905 | 99.8 % / 99.9 % | 0.06 % / 0 % |

(train n = 10000 / dev n = 1000.) The candidate that maximises instruction
following clears each secondary's pool mean on **99.0 % to 100 %** of prompts,
and by a wide margin: 0.599 / 0.634 / 0.652 against pool means of 0.500. Under
ρ at λ = 0 the primary reaches 0.604 and the secondaries 0.570 / 0.592 / 0.604,
all far above the 0.500 reference.

So on this candidate pool maximising the primary objective also raises all three
secondaries well past β times anything they had achieved, the constraints are
satisfied for free, and **MOPO reduces exactly to KL-regularised
single-objective preference optimisation on the primary objective**, for any β
and t0 in the paper's ranges.

This is not a reason to skip the arm: "maximise instruction following and merely
check the others do not fall" is a distinct policy from the four aggregation
rules, and it belongs in the table. It is a reason to report that MOPO's
distinguishing mechanism is inactive here — which is the same absence of
objective conflict that leaves all four aggregation rules indistinguishable,
seen through a different algorithm's diagnostic.

## Still to do

The 80000-row (prompt, candidate) builder, then solve → train → eval.
