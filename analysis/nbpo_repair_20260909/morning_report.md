# NBPO overnight repair — status report

**Run root:** `/work/nbpo_repair_20260909` (pod `nbpo-judge`, namespace `p-aipr`, 4×H200)
**Source:** committed to `exp/iclr27-table1-v2` as `ddc8c2d` (parent `afc527b`)
**Paid judge API calls: 0.** No OpenAI/Anthropic/Gemini/OpenRouter request was made.

*This file is updated as results land. Numbers below are read from run artifacts,
never from a progress bar or a `| tail` exit code.*

## 1. What was wrong, and what is fixed

Eight defects, seven from the audit plus one found during the repair. All are
recorded with code pointers and verification in `audit_findings.json`.

| | Defect | Status |
|---|---|---|
| A | The BT baseline's teacher never reached the training data | fixed — one canonical `log(p*/p_t)` artifact feeds every representation |
| B | Train and validation solved different target problems | fixed — one backend, one global λ = (6.325, 6.186) fitted on train and held fixed on dev/test |
| C | The target-identity check was not an independent solver verification | fixed — ν and Q recomputed from the returned optimizer policy; stationarity 6.7e-11 |
| D | Stage dispatch and config disagreed with what was declared | fixed — one frozen resolved config per arm, hashed, re-checked before the second arm launches |
| E | BF16 logits collapsed the sequence log-probability difference | fixed — FP32 chunked log-softmax/sum, dtypes recorded every optimizer step |
| F | The same candidate was scored under a different prompt depending on its partner | fixed — pool-level tokenization, per-candidate token hashes verified at load |
| G | A substring heuristic ignored `dev` and selected `test` for evaluation | fixed — exact split names, `test` refused outright |
| H | **new** — RoPE `inv_freq` buffers were cast to bf16 with the model | fixed — 33 buffers snapshotted in FP32 and restored, count logged per step |

70 targeted tests pass on the production entrypoints, plus the pre-existing
446-test suite.

## 2. The two matched arms

Everything is shared except the loss: base model and revision, the 2000-prompt
train pool and 500-prompt dev pool (prompt-disjoint, verified: overlap 0), the
frozen preference-model ensemble, the solved `p*`, the tokenization, the
optimizer, and a horizon of 1750 updates at a global batch of 32 pairs fixed
before either arm ran.

- **NBPO-MSE** — the canonical pairwise regression, Eq. (26).
- **NBPO-WBC** — weighted behaviour cloning on the same `p*`, Eq. (28).

## 3. The result that matters so far

The finite-pool target **is** neurally realizable on unseen prompts, and the
weighted-behaviour-cloning objective then walks past it. WBC's held-out dev
diagnostics are monotone in the horizon and change sign:

| updates | nMSE | sign acc. | Pearson | Spearman | mean log-ratio to reference |
|---|---|---|---|---|---|
| 250 | 0.920 | 0.645 | **+0.419** | +0.415 | −0.75 |
| 500 | 1.932 | 0.623 | +0.250 | +0.297 | −2.04 |
| 750 | 3.776 | 0.569 | +0.074 | +0.124 | −3.27 |
| 1000 | 6.121 | 0.535 | −0.028 | +0.018 | −4.39 |
| 1250 | 11.611 | 0.496 | −0.143 | −0.096 | −6.15 |
| 1500 | 12.672 | 0.489 | −0.144 | −0.097 | −6.63 |
| 1750 | 13.070 | 0.486 | −0.152 | −0.105 | −6.68 |

For contrast, the pre-repair pairwise-MSE arms fitted their own training pairs
well and transferred at Pearson ≈ 0 (−0.05) to unseen prompts of the same pool.
A held-out Pearson of +0.42 is the first positive transfer this project has
measured. It is a *diagnostic*, not an acceptance criterion, and it is measured
on dev — no test prompt and no benchmark participated.

The failure mode after that point is not noise: the loss has no stationary point
at the target. Weighted NLL keeps concentrating mass on the highest-`p*`
candidate, the per-response log-likelihood of the whole pool falls by 6.7 nats,
and the induced pairwise log-ratio overshoots the target far enough to
anti-correlate with it.

Because the 1750-update horizon was fixed in advance, that arm stands as the
primary result whatever it shows. A separate 250-update arm was **declared in
writing before being run** (`protocols/wbc_short_horizon_prospective_v1.json`),
labelled dev-selected, and is queued behind the primary evaluation.

## 4. Cost and schedule

| item | value |
|---|---|
| paid judge API calls | 0 |
| WBC arm | 6099.8 s wall, 6.78 GPU-hours, exit 0, 1750/1750 updates on all 4 ranks |

