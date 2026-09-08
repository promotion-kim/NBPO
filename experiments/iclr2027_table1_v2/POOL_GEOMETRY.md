# Pool geometry: 8+8, selected on validation stability and compute

200 SafeRLHF **validation** prompts, responses sampled from one frozen
Llama-3.1-8B-Instruct at identical decoding (T = 1.0, top-p 0.9, 512 tokens).
Learner seeds 1000–1007, comparator seeds 2000–2007 — **disjoint blocks**, so the
two sides are exchangeable by construction. **4+4 is the first four of each side
of the 8+8 draw**, so the comparison is a strict nesting and isolates pool size
rather than a different random draw. 3 200 responses in 65 s.

Scored with the frozen calibrated three-seed GPM and BT ensembles. Each response
is encoded once per (model, seed); all pair scores are head evaluations on cached
encodings, which is why six model-seeds over 200 prompts took 37 s in total.

**No prompted-Qwen artifact is reused.** The quarantined 8×8 pool-size result from
the retired judge plays no part in this.

## The two geometries

| | 4+4 | 8+8 |
|---|---|---|
| responses per prompt | 8 | 16 |
| exact-duplicate pair rate | 0.0073 | 0.0167 |
| mean distinct fraction | 0.975 | 0.918 |
| response length (chars) p10/p50/p90 | 31 / 179 / 2601 | 31 / 174 / 2598 |
| GPM predictive entropy | 0.6117 | 0.6120 |
| GPM ensemble sd (mean / p95) | 0.0429 / 0.0963 | 0.0414 / 0.0960 |
| saturation below 0.02 or above 0.98 | 0.0016 | 0.0012 |
| mean \|margin\| | 0.1549 | 0.1518 |
| GPM/BT sign agreement | 0.9102 | 0.9061 |
| GPM/BT Pearson | 0.9671 | 0.9682 |
| **objective conflict rate** | **0.0767** | **0.0780** |
| pairs per prompt (policy + reference) | 16 + 16 | 64 + 64 |
| **target split-half Pearson** | 0.9172 | **0.9242** |
| **target split-half Spearman** | 0.9132 | **0.9180** |
| **target sign agreement** | 0.8667 | **0.8925** |
| target RMS | 3.414 | 2.336 |
| prompt-wise target variance | 5.979 | 2.977 |
| solver: dual evaluations | 42 | **30** |
| **total solve time (4 solves)** | 24.7 s | **18.3 s** |
| projected KKT residual | 0.0e+00 | 1.8e-14 |
| inner residual | 6.5e-13 | 2.3e-13 |
| Eq. (26) identity residual | 1.8e-15 | 1.8e-15 |

## Selection: 8+8, with no trade-off to adjudicate

Stability first, then compute. 8+8 wins **both**: split-half Spearman +0.0048,
sign agreement +0.026, and it is **faster** — 18.3 s against 24.7 s, on 30 dual
evaluations against 42 — despite carrying four times the pairs. The larger pool
gives the dual a better-conditioned problem, and the per-prompt solves are cheap
enough that four times the pairs does not pay for the extra iterations.

Target split-half is measured by splitting the **ensemble**, not the pool: build
the tensor from one seed, again from the mean of the other two, solve both with
the direct finite-pool solver, correlate the log-ratio targets, averaged over the
three leave-one-out pairings. That asks the question geometry selection should
ask — *if the oracle had been slightly different, would the training target have
been the same?* All halves converged, max projected KKT 3.0e-14.

Downstream policy performance was not consulted.

## Two findings that matter more than the selection

**Generated pools conflict far less than the annotations do.** The
helpfulness/harmlessness conflict rate is **0.077–0.078** on these pools, against
**0.243** on the released human-annotated pairs. That is a threefold drop, and it
is the number the natural bargaining claim actually rests on: a panel built from
same-model samples has much less objective disagreement to arbitrate than the
dataset's headline conflict rate suggests. It should be reported with the
bargaining result, not the 24.3 %.

**Predicted cycles are essentially absent, and the margins are not degenerate.**
On the comparator-vs-comparator reference tournament, the GPM predicts **0 of
800** cycles at 4+4 and **10 of 11 200** (0.0009) at 8+8, against a chance rate of
**0.25**, with a mean absolute margin of 0.154–0.161 on the counted edges. The
margins being far from zero is what separates the two possible readings: this is
a model with real opinions that are almost perfectly transitive, not a model too
uncertain to have any. It agrees with the same measurement on held-out annotated
responses (`SAFERLHF_ENSEMBLE.md`), and it remains a statement about the **model**
— human three-cycles are unobservable here, since the annotation graph has no
triangles at all.

### A bug found in this measurement, and what it looked like

The first version counted cycles on the **policy** block. That block is
`P(learner_i > comparator_j)`: rows and columns index different response sets, so
`M[i,j]`, `M[j,m]` and `M[m,i]` are three unrelated comparisons and "closing" them
is meaningless. It reported a cycle rate of **0.24** — indistinguishable from the
0.25 chance rate, which is exactly what an ill-posed computation over unrelated
signs produces, and which would have read as "the pool is degenerate". The
corrected computation on the comparator-vs-comparator tournament gives 0.0009.
The docstring had described the right block while the code passed the wrong one.

## Artifacts

`pool_responses.jsonl`, `pool_manifest.json`, `probs_{gpm,bt}_{policy,ref}.npz`,
`scoring_manifest.json`, `pool_geometry.json`.
