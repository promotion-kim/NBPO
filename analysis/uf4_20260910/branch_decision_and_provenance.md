# UF-4 branch decision and provenance map

Written 2026-09-11 09:45 KST, after the audit closed and the one-seed comparison
completed. Every number below is measured and cited to its artifact.

## 1. Provenance map

The handoff asks for `labels -> teacher -> target -> checkpoint -> evaluation`
for every panel. There are two panels and they must not be conflated.

### UF-4, the original-rating panel (all policies trained so far)

| Stage | What it is | Artifact |
|---|---|---|
| Labels | UltraFeedback's **released GPT-4 per-completion ordinal ratings**, revision `40b4365`. Pair labels are induced from those scalars, so they are transitive by construction. | `splits/v1/report.json` |
| Teacher fit | ModernBERT-base, four antisymmetric GPM heads, fitted on PM-train 20k / selected on PM-dev 2k by mean NLL. Frozen at step 3750 before any policy existed. | `teacher/gpm_s42`, sha `5f5c8ac6` |
| Policy target | Finite-pool Nash solve, global dual across prompts, beta .25, eta 1, on the 10k policy-train pool | `targets/nash_v1`, solver sha `0ed544c0` |
| Checkpoint | Llama-3.1-8B-Instruct, 1250 updates, seeds 42/43/44 | `arms/nbpo_mse_s4{2,3,4}` |
| Evaluation | Local **Qwen3-14B**, a different model family from the teacher, on the untouched final 2k against one cached base response | `evaluation/final_eval/*` |

### The direct-feedback audit panel (no policy trained on it)

| Stage | What it is | Artifact |
|---|---|---|
| Labels | **New direct pairwise judgments** from Qwen3-14B, 60,000 of them, on 100 reserved prompts disjoint from all five splits | `audit/v1/judgments/audit_100` |
| Teacher fit | none | -- |
| Policy target | none | -- |
| Checkpoint | none | -- |

**The separation that matters:** the trained policies optimize a teacher fitted
to original scalar ratings. The audit is a different feedback source and no
policy is trained on it. A result in the audit does not retroactively change what
the policies were trained on, and this panel stays labelled the original-rating
control.

**Judge independence.** Qwen3-14B supplies the audit judgments *and* evaluates
the final policies. For the UF-4 policy panel that is legitimate: it never
touched the teacher, the targets, or training. If its judgments were ever used
as supervision, it would stop being an independent evaluator for that panel, and
a separate evaluation source would be required.

## 2. Evidence

| Question | Measurement | Verdict |
|---|---|---|
| Does the solver optimize its own objective? | $J$ = -9.5738 Nash vs -9.5757 utilitarian on train; dual residual 3.7e-14 | Yes |
| Does aggregation change the target? | mean TV .0195, 3.44% of target RMS; 24x SafeRLHF's .0008 | Yes, slightly |
| Does the projection realize the target? | dev nMSE .5861 ± .0033 over three seeds, sign .6698 ± .0017 | Partly, better than SafeRLHF's .6845 |
| Are the judge's within-rubric preferences cyclic? | R 0--1 vs fitted-null 0.24--0.97; all Holm $p$ = 1.00; 0 eligible witnesses | No cyclicity separable from noise |
| Why not? | order gap .53--.69 on three of five conditions | Position bias dominates |
| Do the policies improve the four objectives? | helpfulness +.058 / +.048; IF +.017 / +.009; truthfulness and honesty at parity | Only helpfulness decisively |
| Does NBPO beat the matched utilitarian control? | not separated on any of the four objectives | No |
| Is capability retained? | ARC-C, HellaSwag, MMLU all within one harness standard error of base | Yes |

## 3. Decision

The handoff's branch table has four rows. The measurements select the third and
the second together:

- **"Aggregation targets nearly identical -> preserve the control/null result;
  avoid spending three seeds to resolve negligible target differences."** The
  targets differ by 3.44% of target RMS while the projection misses its target
  with nMSE .586, so a policy-level Nash-versus-utilitarian difference is not
  resolvable at this fit quality. The three seeds already committed for each arm
  stand as the null; **no further seeds are spent on this contrast.**
- **"Little or no confirmed cyclic evidence, but objectives conflict and targets
  differ -> proceed with a bounded bargaining comparison; limit claims about
  cyclic-feedback advantages."** Adopted. The within-rubric cyclicity motivation
  is **not** supported on this judge and this data, and the manuscript must not
  lean on it for the UF-4 panel.

Not selected: a new direct-pairwise training pilot. Its precondition was
"meaningful direct-feedback cycles", and the audit found none.

## 4. What the manuscript may and may not claim

May: the solver is certified; the projection transfers; both arms retain
capability; helpfulness improves under an independent judge; aggregation changes
the target measurably but not the policy.

May not: that NBPO beats the utilitarian control; that either dominates the base
in four dimensions; that within-rubric cycles exist in this feedback; that the
absence of detected cycles proves transitivity; that any of this speaks to human
preferences.

## 5. Registered job DAG and remaining cost

Running and queued, all under the persistent controller:

```
util_mse_s43 (running) -> util_mse_s44
each finished arm -> final_eval generation x4 shards -> 4-objective judging
                  -> capability x3 tasks
```

| Remaining stage | Measured unit cost | GPU-hours |
|---|---|---|
| util s44 training | 2.8 h x 4 GPUs | 11.2 |
| final-eval generation, 4 arms | ~25 min x 4 GPUs each | ~6.7 |
| 4-objective judging, 4 arms | 21 min x 1 GPU each | 1.4 |
| capability, 4 arms x 3 tasks | ~25 min x 1 GPU each | ~5.0 |

About 24 GPU-hours remain, against roughly 100 spent. Paid API calls: 0.

Not scheduled, and why: PROSPER and MOPO adaptations need an explicit mapping of
their objectives onto our fixed parties plus a faithful code check, which is not
done; the seven scalarized-DPO points need a BT reward calibration that is built
but unqueued. Both are named in the manuscript as unmeasured rather than
quietly dropped.
