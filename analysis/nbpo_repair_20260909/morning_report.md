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

The whole suite is green against the repaired source: **523 passed, 2 skipped,
0 failed**, of which 70 are the repair's own tests against the production
entrypoints. Two failures on the first full run were the tests being right about
the old code, and both are fixed: the identity test pinned the exponential-map
behaviour that finding C called a defect, and the final-run validator caught the
repository's own `final_iclr2027` config still naming the retired Qwen3-32B judge
as its training teacher. Details in `correctness_test_report.md`.

## 2. The two matched arms

Everything is shared except the loss: base model and revision, the 2000-prompt
train pool and 500-prompt dev pool (prompt-disjoint, verified: overlap 0), the
frozen preference-model ensemble, the solved `p*`, the tokenization, the
optimizer, and a horizon of 1750 updates at a global batch of 32 pairs fixed
before either arm ran.

- **NBPO-MSE** — the canonical pairwise regression, Eq. (26).
- **NBPO-WBC** — weighted behaviour cloning on the same `p*`, Eq. (28).

## 3. The result that matters so far

The finite-pool target **is** neurally realizable on unseen prompts, and it is
the repair that made it so. On 500 dev prompts, prompt-disjoint from the 2000
training prompts (overlap verified 0), at 250 updates:

| Projection | nMSE | Sign acc. | Pearson | Spearman |
|---|---|---|---|---|
| NBPO-MSE, Eq. (26) | 0.839 | 0.692 | **+0.516** | +0.499 |
| NBPO-WBC, Eq. (28) | 0.920 | 0.645 | +0.419 | +0.415 |

Every pre-repair arm fitted its own training pairs and transferred at Pearson
≈ 0 (−0.05) to unseen prompts of the same construction. The paper's own
Eq. (26) works once the implementation is correct; the new loss is not what
unlocked it, and is not credited with it.

For scale, every pre-repair arm in the appendix's diagnostics table sits at test
Pearson between −0.03 and −0.01 with sign accuracy 0.49–0.51 and nMSE 1.02–2.36,
and none passes the gate. The repaired MSE arm at 250 updates meets all four
regression criteria at once (nMSE 0.839 < 0.90, sign 0.692 > 0.65, Pearson and
Spearman both positive) alongside a solver residual of 6.7e−11 against a 1e−4
requirement.

**What that recovery may and may not be attributed to.** The repaired arms differ
from the pre-repair ones in more than the eight defects: the target is canonical
rather than sampled, the reference is a frozen online forward rather than a
cache, the pool is 8+8 drawn at temperature 1.0 / top-p 1.0 rather than 4+4 at
0.9 / 0.95, and the prompt count is 2000 rather than 700. The pool change is
itself part of the repair — a warped sampler makes the occurrence measure
something other than 1/n — but it is also a change of data scale. Two earlier
single-change arms are informative and both were negative: the canonical target
alone at N=700 still transferred at ≈ 0, and the reference fix alone did not move
the held-out regression at 300 updates. So no single fix accounts for this, and
none is credited with it; the claim is about the repaired pipeline as a whole.

The two projections then part company over the horizon. Both first overshoot the
target's own scale — MSE's induced log-ratio RMS goes 1.87 → 2.68 → 3.29 against
a target RMS of 2.37 — but only one of them comes back:

| updates | MSE nMSE | MSE sign | MSE Pearson | MSE drift | WBC nMSE | WBC sign | WBC Pearson | WBC drift |
|---|---|---|---|---|---|---|---|---|
| 250 | 0.839 | 0.692 | +0.516 | −1.13 | 0.920 | 0.645 | +0.419 | −0.75 |
| 500 | 1.226 | 0.688 | +0.452 | −1.86 | 1.932 | 0.623 | +0.250 | −2.04 |
| 750 | 1.482 | 0.678 | +0.419 | −2.42 | 3.776 | 0.569 | +0.074 | −3.27 |
| 1000 | 1.568 | 0.694 | +0.401 | −2.38 | 6.121 | 0.535 | −0.028 | −4.39 |
| 1250 | 1.517 | 0.688 | +0.404 | −2.27 | 11.611 | 0.496 | −0.143 | −6.15 |
| 1500 | 1.432 | 0.691 | +0.412 | −2.12 | 12.672 | 0.489 | −0.144 | −6.63 |
| 1750 | **1.418** | **0.693** | **+0.415** | **−2.07** | 13.070 | 0.486 | −0.152 | −6.68 |

**The two projections do not fail the same way, and only one of them fails.**
MSE overshoots while the cosine schedule is near its peak and then comes back:
its nMSE turns at step 1000 and falls monotonically to the horizon (1.568 →
1.517 → 1.432 → 1.418), Pearson turns with it (+0.401 → +0.404 → +0.412 →
+0.415), the induced log-ratio RMS shrinks back toward the target's 2.37 (3.32 →
3.19 → 3.02 → 2.98), and the drift to the reference contracts (−2.38 → −2.07).
Sign agreement never left 0.68–0.69 and Spearman never left 0.45–0.50, at any
point in the whole run. That is a regression settling onto the stationary point
it has, as the step size anneals.

At the pre-registered horizon the MSE arm therefore meets **three of the four**
criteria — sign 0.693 > 0.65, Pearson +0.415 > 0, Spearman +0.460 > 0 — and
misses only nMSE, at 1.418 against 0.90. At the best single rescaling that
residual is 0.828, so what the gate rejects is magnitude, not direction.

WBC does not turn. It runs monotonically away over the same span, past zero
correlation at 1000 updates and into anti-correlation, with the pool's mean
log-likelihood still falling at 1750. Weighted NLL has no stationary point at
the target, so annealing the step size slows the divergence without reversing
it.

**The residual splits into direction and magnitude, and only the magnitude runs
away.** From the moments each run already logs, the nMSE at the single best
rescaling of $h$ is $1-\E[hT]^2/(\E[h^2]\E[T^2])$; the gap to the reported nMSE
is pure scale.

| updates | MSE nMSE | MSE nMSE* | best scale | WBC nMSE | WBC nMSE* | best scale |
|---|---|---|---|---|---|---|
| 250 | 0.839 | **0.734** | 0.614 | 0.920 | 0.824 | 0.575 |
| 500 | 1.226 | 0.796 | 0.408 | 1.932 | 0.938 | 0.200 |
| 750 | 1.482 | 0.824 | 0.341 | 3.776 | 0.995 | 0.042 |
| 1000 | 1.568 | 0.839 | 0.320 | 6.121 | 0.999 | −0.012 |
| 1750 | — | — | — | 13.070 | 0.977 | −0.046 |

A held-out nMSE of 13.07 reads as total failure and is not what happened: WBC's
*direction* explains 17.6% of the target's second moment at 250 updates and
essentially nothing by 750, while the reported number is dominated by magnitude
the model kept adding after it had stopped adding information. MSE explains
26.6% at 250 and still 16.1% at 1000, and its drift is flattening (−2.42 → −2.38
nats) — which is what a regression with a stationary point at the target should
do, and what weighted NLL does not.

The optimal multiplier being below 1 is shrinkage under imperfect correlation,
not evidence that $\eta$ was set too high — and $\eta$ is *not* the lever here.
nMSE is invariant to rescaling $h$ and the target together, which is what
changing $\eta$ does to first order, so an $\eta$ sweep would move the target's
magnitude without moving this ratio. The quantity that has to shrink is $h$
*relative to* a fixed target, and the levers for that are the ones the 2×2 tests
or regularizes directly: the horizon, the step size, and an explicit proximal
penalty. The horizon is the one being measured tonight.

WBC's full trajectory keeps going until it changes sign:

| updates | nMSE | sign acc. | Pearson | Spearman | mean log-ratio to reference |
|---|---|---|---|---|---|
| 250 | 0.920 | 0.645 | **+0.419** | +0.415 | −0.75 |
| 500 | 1.932 | 0.623 | +0.250 | +0.297 | −2.04 |
| 750 | 3.776 | 0.569 | +0.074 | +0.124 | −3.27 |
| 1000 | 6.121 | 0.535 | −0.028 | +0.018 | −4.39 |
| 1250 | 11.611 | 0.496 | −0.143 | −0.096 | −6.15 |
| 1500 | 12.672 | 0.489 | −0.144 | −0.097 | −6.63 |
| 1750 | 13.070 | 0.486 | −0.152 | −0.105 | −6.68 |

That failure is structural, not noise. Weighted NLL has no stationary point at
the target: it keeps concentrating mass on the highest-`p*` candidate, the
per-response log-likelihood of the whole pool falls 6.7 nats, and the induced
pairwise log-ratio overshoots far enough to anti-correlate with what it was
fitting. The pairwise regression *does* have a stationary point at the target,
so it is not expected to fail this way — its own trajectory is being recorded
and will be reported whatever it shows.

Because the 1750-update horizon was fixed in advance, those arms stand as the
primary comparison. A separate 250-update WBC arm was **declared in writing
before being run** (`protocols/wbc_short_horizon_prospective_v1.json`),
labelled dev-selected, and is queued behind the primary evaluation.

## 4. The bargaining claim, end to end, on held-out prompts

This is the connection the paper has never been able to close. On the 1000-prompt
SafeRLHF held-out split, greedy decoding scored against the same frozen
comparator pools and the same disagreement point used in training:

| | $s_{\text{help}}$ | $s_{\text{harm}}$ | worst $\min_k \E_x[s_k]$ |
|---|---|---|---|
| Base $=\pi_{\text{ref}}$ (greedy) | −0.0405 [−0.0480, −0.0334] | −0.0320 [−0.0394, −0.0244] | −0.0405 |
| \NBPO-WBC, 1750 | +0.0738 [0.0642, 0.0829] | +0.0808 [0.0713, 0.0903] | +0.0738 |
| **\NBPO-MSE, 1750** | **+0.1339 [0.1233, 0.1453]** | **+0.1337 [0.1229, 0.1448]** | **+0.1337** |
| finite-pool solver target $p^\star$ | +0.1653 | +0.1662 | +0.1653 |

Every objective is strictly positive for both trained arms, with paired
prompt-bootstrap intervals that exclude zero — which is the individual-rationality
property Nash bargaining is supposed to deliver, measured on prompts the policy
never trained on. The two objectives also come out balanced (0.1339 against
0.1337 for MSE), which is the bargaining solution's signature rather than an
accident of one objective carrying the other.

Against the finite-pool optimum: the base sits at −0.0405 and the solver target
at +0.1653, so the MSE policy closes **85% of that gap** (WBC closes 56%). The
same ordering holds on dev.

**And it does not cost capability.** Same checkpoints, official evaluators:

| | base | MSE-1750 | WBC-1750 |
|---|---|---|---|
| IFEval strict prompt | 0.7523 | **0.7579** | 0.7412 |
| IFEval strict instruction | 0.8237 | **0.8345** | 0.8177 |
| IFEval loose prompt | 0.7911 | **0.8059** | 0.7800 |
| \textsc{GSM8K} EM | 0.8666 | 0.8643 | 0.8522 |
| \textsc{GSM8K} parse failures | 0.0129 | 0.0136 | 0.0167 |
| median response tokens (\textsc{GSM8K}) | 231 | 227 | 236 |
| HarmBench harmful ↓ | 0.2812 [0.231, 0.331] | 0.2844 [0.234, 0.334] | 0.2562 [0.206, 0.306] |

HarmBench, under the official classifier and the official category routing over
320 behaviours, does not move: all three intervals overlap heavily and neither
arm differs from base. The improvement in the harmlessness *game value* on
SafeRLHF therefore does not show up as a lower harmful-completion rate on an
external safety benchmark, and that is reported as it is rather than folded into
the safety claim. MSE answers harmful requests at 169 median tokens against
base's 60 without being scored more harmful for it.

MSE is at or above base on every IFEval variant and 0.23 points below on GSM8K.
The pre-repair arms lost 15.7 GSM8K points and 9–11 IFEval points on the same
kind of panel. That collapse is gone.

**IFEval's strict checker is not deterministic, by about one prompt.** The same
541 base responses (identical `responses_sha256`) were scored twice by the
official evaluator and disagree: strict prompt accuracy 0.754159 against
0.752311, strict instruction accuracy 0.824940 against 0.823741. That is exactly
one prompt of 541 and one instruction of 834; the loose variants are bit-identical
across both runs. The strict path uses language detection, which is seeded
per-process. So the MSE-over-base strict gap of +0.006 is three prompts against
one prompt of evaluator jitter — real but thin — while the loose gap of +0.015 is
eight prompts and rests on a deterministic check. Both are reported; neither is
averaged away.

**An independent reward model disagrees, and it disagrees with the arm that won
above.** On general-purpose prompts scored by `Skywork-Reward-V2-Qwen3-8B` — a
scalar reward model that took no part in training — the same-prompt win rate
against base is:

| | AlpacaEval prompts (805) | Arena-Hard (500) | Arena creative (250) | median tokens vs base |
|---|---|---|---|---|
| \NBPO-MSE, 1750 | **0.419** [0.387, 0.452] | 0.435 [0.395, 0.476] | 0.390 [0.332, 0.450] | 481 vs 457 |
| \NBPO-WBC, 1750 | **0.578** [0.546, 0.612] | 0.594 [0.552, 0.637] | 0.584 [0.522, 0.646] | 457 vs 457 |

Both intervals exclude 0.5, in opposite directions. So three evaluators order the
two arms three different ways: the training teacher puts MSE far ahead
(+0.134 vs +0.074 worst-objective surplus), the official deterministic
benchmarks put MSE at or above base and WBC below it, and this independent
reward model puts WBC above base and MSE clearly below.

That is not a length artifact — MSE's answers are *longer* than base here — and
it is not a tie that more seeds would resolve. It is the alignment tax showing
up off the training distribution: SafeRLHF is a safety-and-helpfulness panel,
AlpacaEval and Arena-Hard are general assistant prompts, and the arm that moved
furthest toward the SafeRLHF bargaining target is the one a general-purpose
reward model likes least. The reward model is also exactly the kind of scalar
evaluator this paper argues against, so it is reported as a third opinion rather
than as an adjudicator. No Pareto claim survives this table, and none is made.

**What this is not.** These are greedy point-mass decodes, not unbiased samples
from the trained stochastic policy, and the evaluation report says so itself and
declines to certify Algorithm 1 acceptance on that basis. It is one training
seed. The teacher is the same frozen ensemble throughout, and about a third of
what it scores is truncated at its 384-token encoder budget (§6). The test
prompts are the preference model's own test split rather than an untouched
benchmark, which is why the run records `fresh_test_claim` as false. HarmBench is
still missing for these arms — its lane died on a transient CUDA initialization
error and is being re-run.

## 5. Both arms finished, and neither one compresses

| | WBC | MSE |
|---|---|---|
| exit code | 0 | 0 |
| optimizer updates, all 4 ranks | 1750 | 1750 |
| wall time | 6099.8 s | 6730.3 s |
| GPU-hours | 6.78 | 7.48 |
| policy forward tokens (rank 0) | 7,515,461 | 7,515,742 |
| reference forward tokens (rank 0) | **0** | 7,515,742 |
| pre-clip gradient norm, median | 1,076 | 9,274 |
| fraction of updates clipped | **1.00** | **1.00** |
| peak allocated | 74.9 GB | 74.9 GB |

Two things in that table matter beyond bookkeeping. WBC's zero reference
forwards are the runtime proof that its loss never touched a reference, so the
two arms are not FLOPs-matched and are not described as such — the actual
GPU-hours and forward-token counts are reported instead. And *every* update in
both arms was clipped, at pre-clip norms three to four orders of magnitude above
the threshold, which makes each update a fixed-length step along the gradient
direction with the schedule controlling only that length. The audit was explicit
that a large norm is not a reason to raise the clip, and it was not touched.

**Response length is preserved at the full horizon.** Total generated tokens
against base, at the pre-registered 1750 updates:

| benchmark | WBC | MSE |
|---|---|---|
| IFEval | 93% | 97% |
| GSM8K | 106% | 101% |
| HarmBench | 93% | 121% |
| XSTest | 105% | 116% |
| AlpacaEval | 99% | 103% |

The pre-repair arms sat at a median 57% of base length. These do not compress at
all, and their GSM8K chains are slightly *longer* than base. Whether accuracy is
preserved is a separate question that the scoring answers; length was the
mechanism behind the earlier collapse, and it is absent here.

## 6. Why the earlier checkpoints lost usefulness: they compress

A blinded read of a frozen 100-prompt Alpaca subset (slot order randomised per
prompt; every statistic computed before unblinding) settles the question the
audit raised about the pre-repair arms.

- Refusal: **1 of 100** responses opens with a refusal. Over-refusal is not the mechanism.
- Truncation: the arms hit the 2048-token cap on **6** prompts against base's **2** — that is the long tail, not the short one.
- Compression: shorter than base on **80 of 100** prompts, **median 57%** of base length, 57 below 60%, only 7 below 30%.

The answers stay coherent. They keep base's opening sentence and formatting,
drop enumerated items and elaboration, and close with a well-formed summary.
That single mechanism accounts for the whole downstream pattern — IFEval length
and count constraints missed, GSM8K chains cut short, HarmBench harmful rate
down, local RM proxy down.

It is not what the teacher asked for. The solver puts its mass on slightly
**longer** candidates: teacher-weighted mean response length 219.8 tokens
against 205.2 unweighted, and the canonical target is nearly length-independent
(within-prompt $r^2$ of length on the target is 0.006 on train, 0.008 on dev,
0.003 on test). The compression is introduced by the projection.

The alternative explanation — that the teacher asked for shorter, safer answers —
is closed from three directions. Length explains under 1% of the within-prompt
target variance on every split ($r^2$ 0.006 / 0.008 / 0.003). The teacher's mass
sits on longer candidates than uniform (219.8 vs 205.2 tokens on train), and so
does its single highest-mass candidate (222.1). And that top candidate refuses
*less* often than the pool average on all three splits (0.502 vs 0.562 on train,
0.498 vs 0.569 on dev, 0.494 vs 0.540 on test) — under the run's own keyword
heuristic, which is diagnostic-only, so only the within-measure comparison is
claimed. The projection produced something the target did not ask for.

GSM8K splits cleanly: canonN700 loses 15.7 points, of which at most 4.9 can be
parser failure (65 extra unparseable answers out of 1319, counting every one as
otherwise correct). At least 69% is arithmetic that is actually wrong.

One caveat that limits the old comparison: canonN700 and RB1200 emit
byte-identical greedy answers on **44 of 100** prompts, so they were never two
independent arms.

## 7. A standing limit on the teacher

The frozen preference-model ensemble encodes at most 384 tokens, and **26,790 of
the 84,000 pooled response occurrences (31.9%) are longer than that**; on the
evaluation side, 479 of 1,500 (31.9%). Prompt truncation is essentially nil (1
occurrence in 84,000).

For roughly one response in three, then, the teacher scored a prefix. The
finite-pool target carries no information about content past that budget, so it
cannot reward the elaboration the trained arms drop — which bounds how much of
the compression result the teacher could ever have prevented.

It is not a bias toward brevity, and should not be reported as one: the solver's
mass sits on slightly *longer* candidates (teacher-weighted mean 219.8 tokens
against 205.2 unweighted), and the within-prompt $r^2$ of length on the target is
0.006 / 0.008 / 0.003 on train / dev / test. The teacher is frozen for this run
and was not retrained; this is recorded as a limitation, not fixed.

## 8. What the held-out numbers do and do not establish

From the run's own split manifest, stated plainly because it bounds the claim:

- The 2000 policy-training prompts are **all** inside the preference model's
  training set, and the 500 dev prompts are **all** inside its calibration set.
  Dev is held out from the *policy* — which is what the transfer claim needs —
  but it is not held out from the *teacher*.
- The 1000 test prompts are disjoint from preference training and calibration,
  but they are the preference model's own test split and have been scored
  before, so `fresh_test_claim` is recorded as **false**. This is not presented
  as an untouched test set.
- 2 training prompts overlapped a benchmark prompt and were replaced by the next
  eligible prompt in the same hash order; 57 of the 2000 appear in legacy policy
  training. Pretraining and preference-model contamination is not claimed to be
  excluded.

So the correct reading of Pearson +0.52 is: the network generalizes the solver's
target to prompts it never trained on, under a teacher that has seen them. That
is the question the neural-realization bottleneck was about. It is not a claim
about a fresh benchmark.

## 9. Two evaluator faults found, both worth keeping

Neither is about NBPO, and both would have silently degraded the numbers.

**HarmBench could not start its classifier at all.** Two jobs died with "CUDA
driver initialization failed", which reads as a hardware problem and is not one.
`evaluate_responses.harmbench()` loads HarmBench's official `eval_utils.py`
before constructing the vLLM engine, and that module imports torch and touches
CUDA; vLLM v1 then *forks* its engine core, and a forked child cannot
re-initialize CUDA in a process whose parent already has. Reproduced in
isolation: a forked child fails after those helpers are loaded, succeeds without
them, and succeeds either way when spawned. `VLLM_WORKER_MULTIPROC_METHOD=spawn`
fixes it and changes nothing about what is scored. My first re-run assumed the
failure was transient and failed identically — the fresh shell I had tested in
had not loaded HarmBench's helpers.

**The same job then deadlocked after finishing.** Every summary and the
completion marker were written, and the parent sat in `do_wait` on an engine
child that never exits, holding a GPU against the whole queue behind it. The
results were hashed, the engine child was signalled, and the job completed and
recorded `exit_code 0`, `child_exit_code 0`, with its own GPU-idle checks passing.
Because two more HarmBench jobs are queued, this is now automated under
conditions narrow enough that it cannot mask a real failure — the results must
already be complete before anything is signalled — and every signal is logged
with the hashes that were on disk when it was sent.

**And IFEval's strict checker is not deterministic.** See §4: the same 541
responses score 0.754159 and 0.752311 on two runs of the official evaluator,
one prompt of 541, while the loose variants are bit-identical.

## 10. Cost and schedule

**Paid judge API calls: 0.** No request was made to OpenAI, Anthropic, Gemini or
OpenRouter at any point.

GPU accounting from the jobs' own exit records, 24 recorded jobs, **15.89
GPU-hours** in total:

| | wall | GPU-hours | exit |
|---|---|---|---|
| \NBPO-WBC arm, 1750 updates | 6099.8 s | 6.78 | 0 |
| \NBPO-MSE arm, 1750 updates | 6730.3 s | 7.48 | 0 |
| generation, both arms | 266 + 272 s | 0.15 | 0 |
| all scoring lanes | — | 0.65 | 0 |
| HarmBench, two failed starts | 27 + 23 s | 0.013 | 1, 1 |
| profiling runs before the arms | — | 0.84 | — |

The two failed HarmBench starts are listed rather than netted out. The
short-horizon arms and their evaluation are running and are not in this total.

Timeline, all times KST: the pipeline was inherited mid-flight at 16:27 from an
earlier session that lost its connection at 16:02; the WBC arm finished 16:44,
the MSE arm 18:36, generation for both 18:42, the scoring lanes 18:47 with
HarmBench recovered by 19:11, the evaluation controller closed 19:13, and the
short-horizon arms began 19:15.
