# Discrepancy: the handoff's "Appendix C" audit protocol is not in this manuscript

Recorded 2026-09-10 18:05 KST, **before any audit judgment was collected**, as the
handoff requires: "If the local manuscript differs from the specification below,
write the discrepancy before collecting judgments and preserve the currently
authorized manuscript protocol; do not silently combine versions."

## What the handoff asks for

`NBPO_Claude_priority_handoff_20260910.md` §3 specifies a direct pairwise
within-rubric cyclicity audit: 100 reserved prompts plus a 20-prompt pilot, four
reference responses per prompt at T=.8/top_p=.9/2048 tokens, a frozen local
Qwen3-14B judge, four individual rubrics plus a separately labelled joint-rubric
control, two independent panels A and B with ten draws per pair/rubric/panel
split evenly across presentation orders, a core budget of
`100 x 6 x 5 x 20 = 60,000` judgments, A-only cyclic-triangle selection, 200
confirmation draws per selected edge, a fitted scalar-BT tie/position null with
2,000 simulations, whole-prompt bootstrap, and bounded-outcome Hoeffding witness
tests with Holm correction across the four rubrics. It calls this "Appendix C".

## What this repository's manuscript actually contains

`nbpo_iclr/main_v6.tex` (sha256 `aaa98f722453e5f91ba5cacc4a7d0523f787a5bf43390c0cdbb0afd8c640d9c4`,
27 pages, the file the user saved). Its Appendix C is
`\section{Natural Supervision and Preference Models}`, label
`app:natural-supervision`, line 1218. It contains the released-supervision audit
table, the antisymmetric GPM construction, the UF-4 data/teacher/policy
specification, and the aggregation-order discussion.

Measured occurrence counts in that file:

| Protocol element | Occurrences |
|---|---:|
| `Hoeffding` | 0 |
| `Holm` | 0 |
| `C_A` | 0 |
| `R_{\rm BT}` | 0 |
| `panel` | 12, all "SafeRLHF panel" / "UF panel" dataset senses, none a judgment panel |
| `triangle` | 3, all in "zero observed triangles means cyclicity is unidentifiable, not absent" |

**There is no 100-prompt, two-panel, witness-confirmed audit protocol anywhere in
this manuscript.** Nothing was found under a different appendix letter either.

## What the manuscript does authorize, and which I am preserving

- UltraFeedback's score-induced pairwise labels are transitive **by construction**;
  the manuscript states they "do not demonstrate human preference cycles".
- SafeRLHF has zero observed human triangles, so "cyclicity is unidentifiable,
  not absent".
- The generated-pool cycle numbers in Table `tab:pool-pilot` are **model
  predictions**, captioned as such, not human-label measurements.

None of these claims is altered by this discrepancy, and none is being restated
as an audit result.

## Resolution adopted

1. There is no existing protocol in this manuscript to preserve for this audit,
   so nothing is being overwritten or combined. The handoff's §3 specification is
   taken as the authorized specification for the new work.
2. It will be implemented verbatim and written up as a **new, separately named
   appendix section for a direct-feedback audit**, not attributed to the existing
   Appendix C. The existing Appendix C keeps its current content and claims.
3. The distinction the handoff insists on is carried through: the UF-4 policies
   already trained optimize teachers fitted to **original scalar ratings**, and a
   positive Qwen audit does not retroactively make them trained on cyclic labels.
   That panel stays labelled the original-rating control.
4. If the user has a newer manuscript whose Appendix C does contain this protocol,
   it should replace `main_v6.tex` before the write-up, and this file records
   exactly what was implemented in the meantime so the two can be reconciled
   rather than silently merged.

## Consequence for sequencing

Freezing prompt IDs, rubric wording, seeds, parsing and analysis configuration
before observing judgments is unaffected: that freeze is happening now, on CPU,
against the specification above. No judgment has been collected.
