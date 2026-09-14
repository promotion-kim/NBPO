# Review-response audit: September 14 source

Input: `tmp/sep14_source/main_v6.tex` (source status 12:17 KST). The original manuscript was not edited. `sep14_review_updates.json` contains exactly 34 replacement **response contents**, keyed R01–R34; wrap each in the existing `\reviewresponse{...}` macro. Original GPT review contents are separately hashed in `sep14_review_comment_checksums.json` for integration verification.

## Changes that materially update the review status

- R01/R22: the independent target–pool–fresh diagnostic is completed, not pending. Exact-target improvements versus the source coexist with nearly indistinguishable mechanism targets; no advantage of adaptive or Nash aggregation is established.
- R09/R10/R14: actual categorical and fresh measurements now exist. They concern different distributions and valid prompt sets; neither constitutes full-support identification or post-fit population KKT certification.
- R11: credit the reported explicit-pair versus centered-loss value/gradient identity check at 2.8e-14. Keep full trainer verification pending because no training code was supplied. The actual pilot changes pair ordering/grouping; it does not establish candidate-forward computational savings.
- R17: the two projection arms have matched pair multiset, optimizer updates and non-padding response tokens. The 11.2 GPU-hours refer only to one full-run projection, not total cost or all methods.
- R20: use the latest NBPO/base IFEval and HarmBench values; retain capability-regression and protocol questions.
- R21: uniform and two specialist DPO settings now appear; the seven-weight frontier remains incomplete.
- R25: the MOPO row is a plug-in adaptation with robustification removed. RACO is positioned but not experimentally compared.
- R31: nMSE beats zero, while KL worsens and independent pool quality can improve. The 157-prompt paired pilot has no resolved fresh-quality improvement. Target/error RMS comparisons are descriptive, not proof of causal masking.
- R32/R33: natural cyclicity remains unestablished; teacher score co-ranking is not a cycle audit. GPT-4 ratings, human pairwise labels, prompted-judge comparisons and independent evaluation remain distinct sources.

Only formulation/attribution issues R05, R06, R08, R09 and R12 are marked addressed (several explicitly scoped to the formulation). R25 is addressed in positioning with empirical limitations stated. All other issues remain partially addressed rather than promoted to completed on the strength of planned experiments.

## Integration dependencies

The replacements assume root adds Appendix `app:dataset_stress_test` for planned dataset selection / finite-game controls. No other new label is referenced. If this appendix is omitted, replace those references with plain text marking the work as planned. All cited existing labels occur in the source.

## Corrections needed outside review responses

1. The fresh untrained-base statistic 0.4613 differs from min(0.4789, 0.4700, 0.4700, 0.4706)=0.4700. Resolve prompt sets and aggregation. A downward-biased sample minimum does **not** justify replacing the population no-change reference 0.5 with a universal 0.461 offset for every policy; remove that interpretation and the causal dotted-line caption.
2. DPO does not lead every fresh objective: NBPO honesty 0.5221 exceeds DPO 0.5171 in the listed descriptive rows. Unequal row Ns also preclude paired inference from those point estimates.
3. Missing-judgment [0,1] bounds on 200 fixed prompts are identification bounds, not population confidence intervals. Only explicitly supplied objective-wise bounds should support specific claims.
4. The 78-prompt categorical panel and fresh row-specific Ns 186/188/187 cannot establish causal stage transfer. Figure fresh N=174 must be reconciled with its actual common-set contrast.
5. Grouping all 28 pairs for one prompt does not automatically make the averaged gradient seven times larger. The pilot establishes an optimization/batch-composition difference, not its cause.
6. Caption `tab:target-signal` says planned although the table is measured. The capability-details caption still says IFEval/safety pending despite their completed main panel.
7. “Human supervision” must not describe UF's GPT-4 rating-derived labels. The local evaluator revision is declared but not independently traceable to an upstream snapshot.

## Most urgent unresolved review work

On existing artifacts: common-set cross-play recomputation; consistent fresh-statistic definitions; source/manifest verification; matched policy-movement/capability comparisons. On new experiments: fixed-objective, repeatable direct-feedback audit; feasible and meaningfully separated teacher targets; compare exact versus fitted outcomes before expensive multi-seed expansion. Cycle prevalence alone cannot establish a Nash-specific benefit or guarantee feasible improvement.
