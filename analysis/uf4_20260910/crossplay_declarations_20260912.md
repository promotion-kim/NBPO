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
