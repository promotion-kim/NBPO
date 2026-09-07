# Protocol amendment 001 — how judge-protocol evidence is weighted

Status: **binding from this commit forward.**
Written at commit `d3fd455`, on branch `exp/iclr27-table1-v2`.

## What this document is, and what it is not

This amendment fixes how the three families of judge evidence are weighted, so
that the weighting cannot be chosen after seeing which reading is convenient.

**It is not a pre-registration of the Section F holdout.** That run completed at
05:29 PDT on 2026-09-07 and its per-objective metrics were computed, read and
reported *before* this amendment was written. Presenting the rules below as
having preceded that inspection would be false, and the distinction matters
enough to state at the top rather than in a footnote.

Two things below **are** genuine pre-registration, because neither has been
computed or observed at the time of writing:

* the **downstream target-stability analysis** — forward-versus-reverse and
  template split-half tensors, finite-pool targets, and the NBPO /
  Fixed-reference Nash comparisons built from them. None of these quantities
  exist yet in any form.
* the **Section I pool-size pilot** (4+4 versus 8+8), which is running and whose
  outputs have not been read.

The Section F holdout numbers are therefore treated below as *already-known
context*, and the hard decision is deferred to the target-stability gates, which
are still blind. That is the honest ordering, and it is also the reason the
final decision in Section 9 cannot be reached from Section F alone.

## a. Deterministic degradation controls are known-answer, and keep a hard gate

The degradation controls are constructed by a deterministic, objective-specific
edit to a real response. The preferred side is known by construction, not by
anyone's judgement, so disagreement is unambiguously a judge error.

**Gate: hard accuracy >= 0.90 per objective. Retained.**

## b. Identical-response controls are known-answer, and keep a hard gate

A and B are byte-identical. The only defensible verdict is a confident tie;
anything else is position bias with no alternative explanation.

**Gate: confident-tie accuracy >= 0.90 per objective. Retained.**

## c. UltraFeedback natural large-gap labels are pseudo-labels, and are diagnostic only

The natural controls take their preferred direction from UltraFeedback's own
per-aspect ratings, which are **GPT-4-derived pseudo-labels**. Their construct
is not guaranteed to match the frozen v2 rubric — most acutely for honesty,
where our rubric names "pretending to have browsed, verified, cited, observed"
as a failure and the annotation schema does not obviously encode that.

Therefore:

* natural-control agreement is **reported as a diagnostic**;
* it is **not** combined with the deterministic controls into a single
  hard-gate accuracy;
* a disagreement is not evidence of a judge error until adjudicated, which is
  what the blinded honesty review exists for.

This changes an earlier reading: the frozen protocol's "clear-control
directional accuracy" pooled deterministic and natural controls, and its 0.833
on honesty came entirely from natural disagreement while deterministic accuracy
was 1.000. Under this amendment those two are reported separately and only the
deterministic one gates.

## d. Raw position bias stays a mandatory diagnostic; validity is tested downstream

The final estimator queries **both** response orders and averages them, so raw
single-order position bias is not the quantity that determines whether the
measurement is valid — a large but *consistent* order effect is cancelled by
construction, while a small but *erratic* one is not.

Accordingly:

* raw signed position bias, with bootstrap 95% CI, is **always reported**, per
  objective, and is never dropped;
* the **hard validity test** is forward-versus-reverse tensor and downstream
  finite-pool target stability, plus template split-half stability, at the
  thresholds in the governing instruction;
* the raw position-bias threshold is **not** lowered to let a protocol through.
  It is reported at whatever value it takes, and the decision rests on the
  downstream gates.

## e. No protocol choice may depend on method performance

No judge-protocol decision — selection, freezing, adjudication rule, entropy or
order-gap thresholds, pool geometry — may be made by consulting NBPO,
Fixed-reference Nash, BT-RM, Game-KS, Game-utilitarian, PROSPER, MOPO, RACO, or
any policy win rate.

The selection already executed (P2 over P0, P1 and P3) read only calibration and
reliability metrics; `calibration_results.json` records that in
`selection_inputs`. Any future revision must record the same, and a revision
that cannot state its inputs is invalid.

## Consequence for the decision

Because c and d move honesty's natural-label disagreement and raw position bias
out of the hard-gate set, **the protocol cannot be declared valid on the
Section F holdout alone** — favourably or unfavourably. The binding evidence is
the downstream target-stability gates, which are unread at the time of writing.

Until those pass, the protocol is **not** described as validated, and the
7,500-prompt judgment bank stays blocked.
