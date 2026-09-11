# Independent cross-play: declarations recorded before measurement

Written 2026-09-12 02:45 KST. `app:crossplay` requires the evaluator, rubric
wording, parsing, response sampling, held-out prompt IDs, the comparator-weight
prior `a` and the temperatures `beta_k` to be fixed and recorded BEFORE any
comparison. Nothing below is measured yet; this file is the record.

## Prompts: the 3,240 unassigned eligible groups

`splits/v1/report.json`, written 2026-09-09 before any outcome was seen, records
`eligible_groups = 38240` against five splits totalling 35,000 and
`unassigned_eligible = 3240`. Those 3,240 groups were never used to fit the
preference model, train a policy, select on dev, or produce a final-eval number.

Declared: **500** of them, taken in the split builder's own order under
`split_seed = 20260910`, form the cross-play prompt set. Fresh responses are
sampled for every bank policy on those prompts, so no response in this
comparison comes from any training pool.

Using the final-eval 2,000 instead was rejected: those prompts have already
produced numbers I have looked at, and re-using them would make cross-play a
second reading of the same evaluation rather than an independent one.

## Competitor selection: NOT by teacher surplus, and NOT by the final set

Two candidate rules were considered and one obvious one is disqualified.

**Disqualified: pick the arm that won Table 1.** Global game-maxmin is currently
the strongest arm on the final common set. Selecting it as "the strong
competitor" would access final outcomes, which the contract forbids outright.

**Rejected on measurement: the teacher-defined dev surplus.** It is degenerate.
The dev minimum surpluses are nash 0.09116, maxmin 0.09117, utilitarian 0.09108,
fixed-reference 0.09013 -- a spread of about 1e-3 with the top three inside 1e-4
-- and BT-RM's 0.35950 is in normalized-reward units and not comparable to any
of them. A rule that cannot separate its candidates is not a selection rule, and
a rule whose scale differs by arm would pick BT-RM for the wrong reason.

**Declared rule.** The competitor is the arm with the highest minimum-objective
win rate against the base, judged by the frozen independent judge on the
**policy_dev** split (1,000 prompts, never used for final evaluation), among the
trained non-bank arms available when selection runs. Ties beyond 0.002 on the
minimum objective are broken by average win rate, then by arm name. The dev
judging runs at a priority below every main-body training, so it never displaces
a table row, and its result is recorded here before the cross-play bank is
frozen.

## Bank, prior and temperature

Bank: base, NBPO seed 42, fixed-reference Nash seed 42, and the selected
competitor -- exactly the four rows of `tab:crossplay-matrices`.

Declared `a` = uniform over the four bank policies (full support, as required),
shared by every evaluated policy. Declared `beta_k = 0.25` for all four
objectives, the same opponent temperature the adaptive-game arms train at, so
the diagnostic is not additionally tuned per objective.

Diagonal cells stay unmeasured until a self-play sampling and antisymmetry
convention is recorded, per the note in the manuscript source.

## What the bank diagnostic is not

`s_hat^B` regularizes over **comparator weights**, not over response
distributions. It is therefore not an estimate of the population game value or
of population exploitability, and it must not be reported as either, nor
compared against training-teacher surplus. Those are reported separately.

## Cost

Generation: 4 bank policies x 500 prompts. Judging: 6 unordered policy pairs x 4
rubrics x 2 presentation orders x 500 prompts = 24,000 judgments, against 16,000
for one final-eval arm, which took 24 minutes. No paid API is involved; the
judge is the same local Qwen3-14B at revision 40c069824f4251a91eefaf281ebe4c544efd3e18,
temperature 0, both orders, ties 0.5.

## Outcome of the declared selection rule, 07:45 KST

Applied to the two candidates whose dev judging had finished:

| arm | min objective (dev) | average (dev) | n |
|---|---|---|---|
| util_mse_s42 | 0.5074 | 0.5235 | 987 |
| maxmin_mse_s42 | 0.5005 | 0.5182 | 988 |

**Selected competitor: util_mse_s42.** Not within the 0.002 tie band, so no
tiebreak was needed. The rule never read any final-eval number for a non-devsel
arm.

This is worth noting against the objectives table rather than passing over: on
policy_dev the utilitarian arm beats global game-maxmin on the minimum
objective, which is the reverse of the final-split ordering where maxmin leads
every attribute. Two different prompt sets, one seed each, so this is not a
refutation -- but it is a second independent reason not to treat the maxmin lead
as settled before its seeds 43 and 44 land.

## Judge token budget on the cross-play prompts: unexplained

Parse rate of the same frozen judge, same template, same verdict grammar:

| evaluation | prompts | max_tokens | parse rate |
|---|---|---|---|
| final eval | final_eval 2,000 | 256 | 99.4-99.6% |
| dev selection | policy_dev 1,000 | 256 | 99.4% |
| cross-play | unassigned 500 | 256 | 2.6-6.9% |
| cross-play | unassigned 500 | 1024 | 81.8-83.3% |

The failures are cut off mid-reasoning, and the rate is rubric-dependent
(helpfulness 92.4%, truthfulness 86.0%, instruction following 76.2%, honesty
74.5% at 1024). So it is a length limit. What is NOT explained is why these
prompts need several times the deliberation that final_eval and policy_dev
prompts need, given that instruction lengths (median 240 against 186 characters)
and response lengths (median 2,906 against 2,787) are comparable, truncation of
the judge input is zero in all cases, and the comparison type is the same
base-versus-policy in the devsel and cross-play cases alike.

2048 tokens is running. The budget is being raised because raising it
demonstrably raises the parse rate, and this is recorded as an unexplained
difference rather than presented as understood.
