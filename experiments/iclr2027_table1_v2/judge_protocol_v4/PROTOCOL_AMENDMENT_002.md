# Protocol amendment 002 — v4 decoding repair and the evidence it must clear

Status: **binding from this commit forward.** Written before the v4 development
run produced any metric and before `judge_v4_holdout` was opened.

Amendment 001 remains in force: deterministic degradation and identical-response
controls are known-answer and gate; UltraFeedback natural agreement is a
GPT-4-derived pseudo-label reported as a diagnostic only; raw position bias is a
mandatory diagnostic whose validity test lives downstream; no protocol choice may
consult NBPO or baseline policy performance.

## What v3 established, and what it did not

v3 established **two separate defects**, and it matters that they are separate:

1. **A decoding failure.** P2's 96-token budget frequently ended before the
   verdict marker: 4.0-20.9% of pairs per objective had no verdict at all. This
   is a budget problem with an obvious repair.
2. **Insufficient order and template stability among the rows that *were*
   valid.** Confident swap consistency 0.70-0.77 against a 0.85 gate, position
   bias up to +0.19, forward-versus-reverse target Pearson 0.869 against 0.90.

**Repairing (1) does not address (2).** A larger token budget recovers the
missing verdicts; it makes no claim about whether the recovered verdicts are
stable. This amendment therefore forbids describing a larger budget as
sufficient until the fresh holdout and the downstream target-stability gates
both pass on data the protocol has never seen.

There is also a real possibility that fixing (1) *changes* (2) in either
direction — the 13% of pairs v3 discarded were not a random sample, since a pair
that provokes a long rationale is plausibly a hard one. So v3's stability
numbers are not a reliable prior for v4's, and v4's must be measured, not
extrapolated.

## Two candidates, and what separates them

Both carry the same repair: 512-token budget, stop on the first complete verdict
marker with the marker retained, one deterministic 1024-token retry, no
inference of a verdict from a truncated rationale, and no conversion of an
invalid output to TIE.

* **P2-long** — the frozen P2 with *only* that repair. Every semantic field is
  byte-identical, so any change in its verdicts is attributable to budget alone.
* **P2-balanced** — the same, plus a balanced analysis order: t0 assesses
  Response A first, t1 assesses Response B first, with an identical user
  template so wording is not confounded with order. Every pair is run under both
  templates in both presentation orders and `p_hat` is the mean of the four
  semantically-mapped scores, so first-analysed position is balanced by
  construction rather than corrected afterwards.

Both are given two templates and four renderings per pair, so the comparison is
wording variation versus analysis-order variation and **not** budget.

## Accounting rules fixed in advance

* **First-pass invalid** and **final unresolved invalid** are stored and reported
  separately. The gate is on final unresolved; the first-pass figure records how
  much work the retry is doing, which is the evidence for whether 512 is the
  right budget or merely a lucky one.
* **Verdict-token position** percentiles are reported. A p99 close to the cap
  means the budget was chosen too tightly even if the invalid rate passes.
* **No adaptive omission** of templates during development or holdout. Every
  pair gets both templates and both orders, so the template split-half is
  measurable on the whole set rather than on the adjudicated subset — which is
  what made v3's template split unmeasurable.
* **Complete tensors only** for target stability, with complete-prompt coverage
  reported and gated at 0.99. v3's order-split estimate rested on 24 of 200
  prompts because incomplete tensors were dropped; an estimate from the residue
  of a broken run is not evidence about the method.

## Eligibility before ranking

A candidate is eligible only if, on **every** objective: final unresolved
invalid < 0.2%, deterministic degradation >= 0.90, identical confident-tie
>= 0.90. Ranking is consulted only among eligible candidates. If none is
eligible the outcome is "no eligible protocol", not the least-bad one.

## What a pass does and does not license

Passing every hard gate licenses launching the full judgment bank with v4 labels
only. It does **not** retroactively validate v2 or v3 rows, which stay
quarantined and unmixed, and it does not license reusing the quarantined v3
pool-size result to choose 4+4 versus 8+8 — that comparison restarts on a new
development set under the selected v4 judge.
