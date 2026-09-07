# STATUS — ICLR-2027 Table-1 rebuild

Updated 2026-09-08, session 5. Branch `exp/iclr27-table1-v2`, pushed to `nbpo` (promotion-kim/NBPO).

Nothing below is a projection. Runs that are still going are named as such, with
their log path, and no number is quoted from them.

## Gate summary

| gate | state |
|---|---|
| baseline suite unchanged | **pass** — 219 → **328 passed**, 2 documented skips, 0 failures |
| generic solver == audited solver on adaptive-game Nash | **pass** — bitwise on lambda, pi, surplus, d and every residual |
| finite-pool target identity | **pass** — residual ≤ 1.8e-15 on all six matched rows |
| eta applied exactly once | **pass** — pairwise target scales linearly in eta |
| split prompt-disjointness | **pass** — all six exact-overlap checks zero, re-derived from the written files |
| benchmark decontamination | **pass** — 8 exact + 2 approximate removed against all five benchmarks |
| Game-KS properties | **pass** — 14 property tests incl. an independent grid-search check |
| 50-prompt smoke: pools | **pass** — 8 response files, 2 fingerprint-bound manifests |
| 50-prompt smoke: judging | **pass** — 8800 calls, 0 % invalid, 0 retries |
| judge audit: parse rate | **pass** — 0.000 % invalid (gate < 0.2 %) |
| judge audit: cell completeness | **pass** — every semantic pair judged in both orders |
| judge audit: **swap consistency (v2)** | **FAIL** — 0.59–0.70 against a 0.85 gate. Diagnosed, not lowered; v2 is retired. |
| v2 swap mapping correct | **pass** — six hand-constructed cases + exact skew symmetry |
| v3 calibration controls built | **pass** — 800 controls, 800 distinct prompts, zero split/benchmark overlap |
| v3 candidate protocols calibrated | **pass** — P0/P1/P2/P3, adaptive and all-templates passes |
| v3 protocol frozen | **pass** — P2_deliberative, 2 gates recorded open at freeze |
| **Gate 1: invalid rate** | **FAIL** — 4.0–20.9% per objective against 0.2%. Missed at calibration; see below |
| **Deterministic-degradation gate** | **P2 passes** (0.96–1.00); P0/P1/P3 fail on honesty and truthfulness |
| **Identical-pair confident-tie gate** | **pass** — 1.000 for every protocol and objective |
| **Section F holdout (200 prompts)** | **FAIL** — confident swap 0.70–0.77 against 0.85; position bias up to +0.19 |
| **Downstream target stability** | **FAIL** — target Pearson 0.869/0.870 < 0.90, sign agreement 0.736/0.771 < 0.85 |
| Admissible judge protocol | **NONE**, and the route is now **closed**. See the v5 factorial below. |
| **v5 opaque-identifier factorial** | **Decision B** — both identifier schemes fail; prompted-Qwen judging retired |
| **Controlled-nontransitivity feasibility (rho\*)** | **pass** — rho\* > 0 on all 25 v1 instances, certified gap ≤ 3.1e-7 |
| **Exact NBPO on the stress test** | **pass** — exact global and proximal Nash reach 88–94 % of rho\* at every alpha |
| **Deployed alternating solver** | **FAIL** — fixed-point residual 1.000 at alpha ≥ 0.5; the stress-test negative was this |
| **SafeRLHF splits** | **pass** — prompt-disjoint, overlap 0, hashes recorded |
| **SafeRLHF observable cycles** | **ZERO, structurally** — the annotation graph is a near-perfect matching |
| **GPM beats a constant predictor** | **pass** — NLL 0.56/0.52 vs 0.66/0.60, balanced accuracy 0.72/0.75 vs 0.50 |
| **GPM beats BT** | **no** — ΔNLL −0.0039 [−0.0073, −0.0005] on helpfulness only; no accuracy gain on either |
| Full 7,500-prompt bank | **NOT LAUNCHED**, and blocked |
| Section I pool-size pilot (4+4 vs 8+8) | **QUARANTINED** — stopped mid-run, `exploratory_v3_invalid_protocol`; may not select pool geometry |
| v4 decoding repair (512/1024, stop-on-marker) | **pass** — final unresolved invalid 0.0000 on every objective |
| v4 known-answer eligibility | **P2-long eligible and selected** (invalid 0.0000, deterministic 0.960, identical 1.000); P2-balanced **rejected** — deterministic 0.800 on honesty and truthfulness |
| **v4 stability — confident semantic swap** | **FAIL** — 0.412–0.708 against 0.85, every objective |
| **v4 stability — order split-half Spearman** | **FAIL** — 0.189–0.428 against 0.75, every objective |
| v4 real-pair position bias | **severe and concentrated** — +0.364 helpfulness, +0.404 instruction following; +0.037 / +0.016 on honesty and truthfulness |
| **Admissible final judge protocol** | **NONE.** P2-long is selected among candidates and still not admissible |
| judge_v4_holdout / backup holdout | **unopened / sealed** |
| 50-prompt smoke: tensors | **pass** — measured `d`, exact skew-symmetric reference tensor |
| 50-prompt smoke: solves | **pass** — 4 matched rows, identity residual ≤ 3.6e-15, matched ‖w‖₁ and KL |
| regression-realization gate | not started (needs the pilot) |
| final seeds 42/43/44 | not started |

## What is done

**Solver and baselines.** Objective representation and aggregation are now
independently selectable, sharing one KL-proximal solve, one target artifact and
one realization path:

* `adaptive_game` — the manuscript's finite-temperature game;
* `fixed_reference` — comparator frozen at the empirical `mu`; verified
  numerically to be the `beta -> infinity` limit of the adaptive game, which is
  what makes "NBPO vs Fixed-reference Nash" isolate the adaptive opponent.
  Implemented in `scripts/nbpo`, **not** by calling the legacy `scripts/bpo`;
* `bt_reward` — scalar Bradley-Terry heads on the same judged pairs;
* aggregations `nash`, `utilitarian`, `kalai_smorodinsky`, plus the two legacy
  max-min controls.

Game-KS is the real bargaining rule: an individually rational ideal point, the
ideal-**normalized** egalitarian stage, a lexicographic Pareto refinement, and a
proximal selection whose achieved divergence is reported. On the verification
pool it equalizes the normalized surpluses to within 5e-3 and reaches a minimum
normalized surplus of 0.512 where raw surplus max-min reaches 0.495 — the two
are measurably different rules, which is the point.

**Data.** One deterministic prompt-level UltraFeedback split (7000/500/1000,
seed 20260907) built by grouping near-identical prompts *before* splitting —
4343 near-duplicate pairs turned out to need merging. Rubric v2 is a new file;
v1 is untouched.

**Infrastructure.** A registry-backed, resumable launcher with a phase-aware GPU
policy and exit sentinels, so `status` stays honest after a crash.

## Running now

Nothing is running on the pod's GPUs as of this update; all three are idle.

Completed on the pod: both response pools (02:46–02:49 PDT, one GPU) and the
50-prompt training judgment bank (02:51–02:57 PDT, Qwen3-32B TP=2) —
8800 rows at `/work/iclr27_table1_v2/smoke/verdicts.jsonl`; the v1-rubric
diagnostic (03:03–03:16, aborted by design); and the Llama-3.3-70B cross-judge of
the same pairs (03:17–03:28) at `/work/iclr27_table1_v2/diag/verdicts_v2_llama70b.jsonl`.

## Judge protocol v3 — calibration (Sections A–E)

**The v2 swap mapping is correct.** Before attributing the v2 failure to
anything, the arithmetic that turns two ordered verdicts into one semantic
preference was pinned against the six hand-constructed cases: forward-A /
reverse-B → 1.0, forward-B / reverse-A → 0.0, both position contradictions →
0.5, tie/tie → 0.5, and the reference tensor exactly skew-symmetric with a zero
diagonal and no projection needed. A test that the reverse verdict is genuinely
flipped fails loudly against the obvious wrong implementation. So the failure
was the instrument, not the arithmetic.

**Calibration controls.** 800 pairs with known ground truth over 800 distinct
prompts, disjoint from every split and all five benchmarks (re-derived from the
written file): 400 natural pairs with a ≥3-point gap in that objective's *own*
UltraFeedback per-aspect rating, 200 byte-identical pairs whose only defensible
verdict is a confident tie, and 200 deterministic objective-specific
degradations. Slot assignment is randomized, so a position-biased judge scores
at chance rather than perfectly.

Manual inspection of the first build caught two defects and both were fixed: the
instruction-following degradation was cut mid-list by a regex sentence splitter,
and every family drew from the top of one shuffled list so 800 controls spanned
~150 prompts.

**Result of the first (adaptive) calibration pass**, worst objective:

| metric | P0 hard | P1 soft logits | P2 deliberative |
|---|---|---|---|
| min clear-control directional accuracy | 0.460 | 0.540 | **0.852** |
| honesty degradation accuracy | 0.060 | 0.020 | **1.000** |
| truthfulness degradation accuracy | 0.220 | 0.340 | **0.875** |
| identical-pair confident-tie accuracy | 1.000 | 1.000 | 1.000 |
| max \|position bias\| | 0.275 | 0.230 | **0.035** |
| confident-pair swap consistency | 0.548 | 0.608 | **0.953** |

Two findings, and they point in different directions.

*Soft logits alone do not help the judge apply the criterion.* P1's honesty
degradation accuracy is **0.02** — worse than the hard baseline — and it fails
confidently: on 49 of 50 pairs it prefers the response opening "I just browsed
the official documentation and personally verified every claim below", with zero
tie mass and Δ ≈ +0.4. Nothing in the transcript proves that claim false, but the
v2 honesty rubric names "pretending to have browsed, verified, cited, observed"
as a honesty failure, so this is the judge not applying the stated criterion.

*The explicit decision procedure is what repairs it.* P2 takes honesty
degradation to 1.000 and cuts worst-objective position bias by a factor of eight.

But P2 is a hard verdict: its entropy is identically zero, so it cannot express
uncertainty — which was the whole reason for leaving v2. **P1 and P2 repair
different halves of the problem**, so `P3` composes them in two exact stages:
generate the one-sentence justification, then score the three verified
single-token labels at the position after it, with the rationale in context.
P3's calibration run is **in flight**; no P3 number is quoted here.

A tokenization note that was not a formality: under the Qwen3 tokenizer `TIE` is
**two** tokens and `[[A]]` is three, so scoring those strings as single
candidates would have been silently wrong. The verified single-token sentinels
are `A`=32, `B`=33, `T`=51, re-checked in context at load time and mapped back to
A/B/TIE in every artifact.

## Two analyzer bugs of mine, found and fixed

Both inflated how good things looked, so they are stated before the results.

1. **The invalid rate was computed over a list already filtered to valid rows**,
   making it identically zero. My previous report of "invalid 0.000%" on the
   holdout was wrong: the true rate is **13.38% of pairs** (truthfulness 20.9%,
   instruction following 15.5%, honesty 13.1%, helpfulness 4.0%), because a
   deliberative rationale can exhaust the 96-token budget before the verdict
   marker.
2. **The calibration reported invalid rate only in a run-level summary** that
   neither the ranking line nor the selection rule consulted. P2 was already at
   **1.0%** there — five times the 0.2% gate — and I froze it anyway. Gate 1 is
   now a hard *prerequisite* in the selection key, reported per objective.

A third, smaller one was fixed last session: the order split-half correlation
was computed with a double sign flip and came out negative.

## No admissible protocol

With gate 1 enforced and Amendment 001 separating deterministic from natural
controls, the picture is:

| protocol | max invalid rate | min deterministic-degradation accuracy |
|---|---|---|
| P0 hard verdict | 0.000 | **0.14** (honesty) |
| P1 soft logits | 0.000 | **0.00** (honesty) |
| P2 deliberative | **0.035** | 0.96 |
| P3 deliberative + soft logits | 0.000 | **0.48** (honesty) |

**Every candidate fails a known-answer prerequisite.** The three protocols that
always parse are exactly the three that cannot detect a deterministic dishonesty
or falsehood edit, and the one that detects them is the one that sometimes runs
out of budget saying why. That is a mechanism, not a coincidence: reasoning
before answering is what catches the edit, and generating it is what risks the
token budget.

The calibration analyzer now refuses to name a winner in this situation rather
than ranking the least-bad candidate.

## Downstream target stability — FAIL

Two tensors from disjoint halves of the same observations, mapped to one
semantic orientation, each solved with the identical finite-pool procedure.

| split | method | target Pearson | sign agreement | policy TV median / p90 | rank agreement |
|---|---|---|---|---|---|
| order | NBPO | **0.869** | **0.736** | 0.224 / 0.705 | 0.517 |
| order | Fixed-reference Nash | **0.870** | **0.771** | 0.190 / 0.464 | 0.450 |
| template | both | **not measurable** | — | — | — |

Gates are 0.90 and 0.85; both fail. Only **24 of 200 prompts** had a complete
tensor in both order halves — a consequence of the 13% invalid rate — so the
sample is small and the estimate is correspondingly weak; that is itself a
symptom rather than a mitigating detail.

The template split is **not measurable at all**: only adjudicated pairs receive a
second template, and a complete tensor needs every cell of a prompt adjudicated,
which never happens. Measuring it requires an all-templates holdout.

## Section F — the frozen protocol fails on real pairs

17600 pairs over 200 held-out validation prompts (disjoint from train, test and
the calibration controls), 56548 judge calls, 24 minutes. The gates fail, and
the *shape* of the failure is the finding:

| objective | confident swap: controls → real | position bias: controls → real |
|---|---|---|
| helpfulness | 0.973 → **0.727** | +0.013 → **+0.156** |
| instruction following | 0.917 → **0.703** | +0.062 → **+0.193** |
| honesty | 0.927 → 0.774 | +0.013 → +0.049 |
| truthfulness | 0.891 → 0.772 | −0.009 → +0.040 |

Same frozen protocol, same judge, same rubric — only the pairs differ. **The
judge is reliable when a real difference exists and unstable when it does not.**
That is Section I's branch, reached by measurement rather than assumption.

Also failing: unresolved uncertainty 0.25–0.32 against a 0.25 ceiling, and
split-half Spearman below 0.75 everywhere. The template split (0.42–0.44) is
measured only on the adjudicated subset — by construction the unstable pairs —
so it is biased low; the order split (0.54–0.62) is unbiased and still short.

Cycles are low on reliable edges (0.000–0.016), BT deviance 0.07–0.09. So
intransitivity is *not* what drives this, consistent with the earlier finding
that these criterion judges are close to transitive.

One number in the first run of this audit was a bug of mine, not a judge result:
the order split-half came out negative, which is impossible. `semantic_score` is
already learner-oriented in both orders and I had written the reverse estimate
as `0.5 - score`, double-flipping it. Fixed and pinned with a test.

## Decision

**Branch C, with a gate-1 failure on top.** Deterministic controls fail for P0,
P1 and P3 (honesty and truthfulness); P2 passes them but fails the invalid-rate
gate and downstream target stability. Therefore:

* the 7,500-prompt judgment bank is **not launched** and stays blocked;
* no RM or policy training is launched;
* no v2 label is reused anywhere;
* the protocol is **not** described as validated.

The obvious repair is a larger judge token budget for P2 — a decoding change,
not a semantic one, leaving rubric, templates and decision procedures untouched.
It nonetheless requires re-freezing and re-running calibration and a *fresh*
holdout, because the current holdout has been read.

## Section I — pool-size pilot: QUARANTINED, not running

The 8+8 arm was stopped cleanly ~19 minutes in on operator instruction and the
whole directory moved to
`QUARANTINED_poolsize100_exploratory_v3_invalid_protocol`. Its results file was
never written, so no partial score exists; pools and pair lists are preserved.

**It may not be used** to select 4+4 versus 8+8, to fill any paper cell, or in
combination with v4 measurements. The pool-size comparison restarts from scratch
on a new development set once a judge passes.

## The solver leg, on the real judged bank

The four game rows were solved on the 50-prompt bank (4000 dual iterations,
R = 3, beta = 0.25, eta = 1.0). Every one reproduces its own weighted-q target
to 1e-15, which is the identity the Eq. (26) pair builder depends on.

| row | min s | sum s | KL | ‖w‖₁ | identity |
|---|---|---|---|---|---|
| NBPO (adaptive + Nash) | 0.0886 | 0.5338 | 0.768 | 31.71 | 3.6e-15 |
| Fixed-reference Nash | 0.1056 | 0.5886 | 0.822 | 28.38 | 1.8e-15 |
| Game-utilitarian | 0.0886 | 0.5425 | 0.821 | 31.71 | 3.6e-15 |
| Game-KS | 0.0909 | 0.5358 | 0.811 | 31.71 | 1.8e-15 |

The three adaptive-game rows sit at the same weight norm by construction, and
their proximal divergences land within 0.05 of each other, so they differ in the
direction of the weight vector rather than in step size. Fixed-reference carries
its own Nash norm, as it must.

The disagreement point is **measured**, and the two representations disagree
about it exactly as the theory says they should: the adaptive game gives
`d = [-0.043, -0.034, -0.030, -0.065]` (a soft-min of a skew game at finite beta
is negative), while the fixed-reference control gives `d = [0, 0, 0, 0]` — which
is what `V^FR(mu)` evaluates to on an exactly skew-symmetric shared pool. The
code computes it in both cases; neither is hard-coded.

Game-KS behaved as a bargaining rule: ideal point individually rational on all
four objectives, IR violation exactly 0, `rho* = 0.822`, normalized surpluses
0.829–0.870, Stage-2 refinement applied with a positive gain. The spec-literal
diagnostics at a near-unregularized step agree (`rho* = 0.828`, IR violation 0).

These are pipeline-validation numbers on 50 prompts. **They are not results and
must not reach a table.**

## The swap-consistency gate fails, and what the diagnosis says

On the 50-prompt smoke bank the swap-consistent winner rate — agreement over the
pairs **both presentation orders decided** — is 0.696 / 0.663 / 0.645 / 0.592
for instruction-following, truthfulness, helpfulness and honesty. The gate is
0.85. It has not been lowered, and no result is being reported through it.

What has been ruled out so far, each by measurement rather than by argument:

* **Parsing.** 0 of 8800 calls failed to parse and none needed a retry. The
  regex takes the last `[[A]]/[[B]]/[[TIE]]` marker, and `to_policy_win` maps
  the shown order back to the learner's point of view; both were re-read.
* **Position bias.** +0.145 for helpfulness and within ±0.04 for the other
  three. A judge that simply favoured the first slot would show +1.
* **Thinking mode / truncation.** `enable_thinking: false` is applied *and*
  recorded, and every completion contains a verdict marker inside 16 tokens.
* **The rubric.** Re-judging the identical pool with the **v1** rubric at the
  same 16-token budget fails outright: 86.5 % unparseable on the first pass and
  **6684 of 8800 cells still invalid after two retries**, at which point the
  judging CLI refused to write a matrix rather than emit a bank with holes in
  it. v1's phrasing makes the judge write a preamble, and the published
  campaigns always ran it at 512 tokens. v2 is 0 % invalid at 16 tokens. So v2
  is strictly better on the axis it was designed for, and the consistency number
  is not a v2 artifact.

The live hypothesis is the **pool**, not the judge: eight samples from one 8B
model on one prompt are frequently near-equal, and a judge asked which is
*materially* better on a narrow criterion has little to latch onto. The tie
rates say the same thing — 0.40 to 0.66 swap-averaged, and 580 of 1100 honesty
pairs are ties in both orders. Under that reading the low consistency is the
measurement being honest about indifference, and swap averaging is doing exactly
the job it exists for.

That hypothesis was tested directly: the same pairs, the same rubric, the same
decoding, judged by **Llama-3.3-70B** (`/work/models/xj_judges/llama70`, TP=2,
`--judge-role monitoring` so it can never be mistaken for the training bank).
The result rules against blaming Qwen3-32B.

| objective | Qwen3-32B | Llama-3.3-70B | both-order-decisive pairs (70B) |
|---|---|---|---|
| truthfulness | 0.663 | **0.807** | 176 of 1096 |
| honesty | 0.592 | **0.710** | 162 of 1100 |
| instruction following | **0.696** | 0.571 | 161 of 1100 |
| helpfulness | **0.645** | 0.433 | 527 of 1100 |

**Neither judge reaches 0.85 on more than one objective, and the 70B does not
reach it at all.** It is better where it is much more cautious — it ties both
orders on 762 of 1100 honesty pairs and 734 of 1096 truthfulness pairs, leaving
only ~160 decided — and it is *worse* on the other two. Helpfulness at 0.433 is
below chance: on that criterion the 70B contradicts itself more often than it
agrees whenever both orders commit.

So the failure is not a property of the training judge, and swapping judges will
not fix it. Two things follow:

1. the reading that the **pool** is the cause survives the test — near-equal
   samples from one 8B model give a judge little to be stable about, and both
   judges respond by tying heavily;
2. **helpfulness is the worst-behaved criterion under both judges** and needs
   attention on its own terms before the full bank is labelled, rather than
   being averaged in with the other three.

The 70B also sits at 0.170 % invalid with retries needed on 10 cells — inside
the 0.2 % parse gate, but only just, where Qwen3-32B was at 0.000 %.

The full labelling run stays blocked. The honest options are to fix the
measurement (a pool with real quality spread, e.g. comparators from a different
checkpoint, which would also make the surpluses less degenerate) or to revise
the 0.85 threshold deliberately and on the record with these numbers attached.
Neither is a decision to take silently, and neither has been taken.

## Findings worth carrying forward

1. **Matched step size is a real confound.** BT-RM-utilitarian solved at the
   *game's* weight norm instead of its own gives KL 1.343 against BT-RM-Nash's
   0.996 — a 35 % larger step that would read as a method effect. Each
   representation matches to its own Nash `||lambda||_1`; `--match-weight-l1-to-nash`
   enforces it.
2. **A literal three-stage KS has no neural realization.** Over the raw simplex
   the Stage-3 argmin-KL point sits on a face of the simplex, so its log-ratio
   target is unbounded. The stages are therefore solved inside the eta-proximal
   family, and the spec-literal quantities are still computed and written by
   `ks_unregularized_diagnostics` so the deviation is measured, not asserted.
3. **A positive ideal point is not enough.** On a zero-sum pool the individually
   rational set collapses to `{s = 0}` and `u_k` is positive only by float noise
   (~1e-13); dividing by it yields a `rho*` that looks like a solved problem. The
   solver now judges `u_k` against the attainable range and refuses.
4. **Individual rationality can genuinely fail** at a small step even on a
   well-formed pool. That is reported as a typed error with diagnostics, never
   clamped.
5. **`fixed_reference` and `bt_reward` make `R` meaningless** — their Eq. (21)
   map is constant, so one application lands on its fixed point.
   `extra_map_residual` is exactly 0 at every `R`; `fixed_point_residual` is the
   last-iteration *change*, so it is nonzero at `R = 1` by definition. Both are
   reported as-is.
6. **UltraFeedback is more redundant than the pair-level split implied**: 552
   multi-prompt groups, the largest with 103 members.
7. **A 16-token judge budget is only viable with the v2 rubric.** v1 at 16
   tokens is 86.5 % unparseable; it was always run at 512. Anyone re-running a
   published panel must keep v1's 512-token budget.
8. **The bank now keeps the raw completion.** It did not, and the first question
   a bad consistency number raises — what did the judge actually say? — was
   unanswerable without re-running everything.
9. **Objectives agree more than they conflict here**: pairwise correlations of
   the swap-averaged margins run +0.56 to +0.73, and hard directed-cycle rates
   are 0.000–0.0025 at the predeclared 0.1 margin. Consistent with the earlier
   finding that these criterion judges are close to transitive, so any
   bargaining effect has to come from magnitude disagreement rather than from
   intransitivity.

## Blockers

**The four H200s of `nbpo-judge2` do not exist to be used.** That pod requests
one GPU and has never scheduled; both usable nodes in its zone are fully
allocated and the zone's other two are cordoned. On the operator's instruction
the campaign runs on the three idle H200s inside the running `nbpo-judge` pod, so
judge phases are TP=2 and the policy fan-out is three-wide. This lengthens the
schedule; it does not change any scientific configuration.

**Disk is the tightest resource.** Existing campaigns hold ~8.5 TB on the PVC.
Only the fixed final iterate is retained per run, and only after its hashes and
evaluation artifacts are complete.

## Next

1. **Resolve the swap-consistency gate.** It is the only thing blocking the
   expensive phases, and it is now known not to be fixable by changing judges.
   The two honest routes are above; both need a decision rather than a rerun.
2. the rest of smoke step 9 — pair targets → short 8B fit → decode → independent
   evaluation. The solver leg is done; the pair-target handoff is fixed and
   tested but has not yet been run on the real artifact;
3. 200-prompt judge audit once the gate question is settled (step 10);
4. full train/validation pool and judgment bank (step 11);
4. beta/R diagnostics, then the sequential eta/LR/step pilot on tuning seed 11
   (steps 12–13);
5. regression-realization gate, then freeze and launch seeds 42/43/44.

RACO is **not** started: `references/chen_raco_icml_2026.pdf` is present but no
official implementation has been located, and the brief forbids inventing one.
It stays marked blocked until a faithful source is pinned.

## v4 development — FINAL (2026-09-08)

Full record: `judge_protocol_v4/DECISION_V4_FINAL.md`; numbers in
`judge_protocol_v4/development_final_{report.md,metrics.json}`.

**P2-long is selected and is not admissible.** It is the only candidate that
passes all three known-answer prerequisites (final unresolved invalid 0.0000,
deterministic degradation 0.960 min, identical-pair confident tie 1.000), and it
fails both pre-registered natural-pair stability gates on every objective:
confident semantic swap 0.412–0.708 against 0.85, order split-half Spearman
0.189–0.428 against 0.75. P2-balanced is rejected earlier, on the known-answer
controls (deterministic 0.800 for honesty and for truthfulness), and its position
bias is worse, not better.

**The decoding problem is solved.** Measured verdict-token positions have p99 at
98–151 and a maximum of 333 against v3's 96-token cap — the mechanism behind the
4–21 % invalid rate, confirmed directly. At 512 tokens with a 1024-token
deterministic retry, final unresolved invalid is 0.0000 on every objective.

**The instability is concentrated.** Helpfulness and instruction-following carry
+0.364 / +0.404 position bias with swap consistency *below chance*; honesty and
truthfulness carry +0.037 / +0.016 and reach ≈0.70. One bounded diagnostic
remains: an opaque-identifier factorial that separates the lexical "A" effect
from the physical first-slot effect, which the ordinary rendering confounds
exactly. If it fails, the prompted-Qwen training-judge route is retired for the
main paper; no further prompt variants will be made either way.

**A pairing bug found while freezing this.** `v4_dev/scored_controls_P2_long`
was judged by the 2-template cost-parity cut (`ac8fcab2bf6d`), not by the
restored 3-template protocol (`33962ad4275f`). Scoring the 3-template dev run
against those controls moves truthfulness deterministic accuracy 0.960 → 0.860,
across a hard gate. `analyze_v4.py` now derives each results file's sibling
manifest and refuses to mix protocols; two tests pin it.

Unchanged and still true: `judge_v4_holdout` unopened, backup holdout sealed, the
7 500-prompt bank not launched, no RM and no policy trained from any v2/v3/v4
prompted-Qwen label.

---

# Session 5 (2026-09-08): three bounded tracks

## Track 1 — the prompted-judge route is closed

Full record: `judge_protocol_v5/DECISION_V5_FINAL.md`.

A single factorial on `judge_v4_dev` (24 000 renderings, 1 023 s, 600 per cell,
balance re-derived from the written file) separates two things the ordinary
rendering confounds perfectly. **The bias is lexical.** Under A/B the physical
slot contributes +0.012 on helpfulness and +0.026 on instruction following while
the letter "A" contributes **+0.237** and **+0.147**. It is strong enough to
override a known answer: deterministic-degradation accuracy in the factorial's
A/B condition is 0.480/0.680/0.980/0.640, and on the same controls under opaque
identifiers it is 1.000/1.000/1.000/0.940.

**Opaque identifiers relocate the bias, they do not remove it.** The total
first-and-"A" advantage is conserved (0.249 → 0.236 helpfulness, 0.173 → 0.153
instruction following); the mass simply moves into the physical channel (+0.168,
+0.158). Both schemes fail confident semantic swap (0.448–0.750 vs 0.85) and
order split-half Spearman (0.008–0.465 vs 0.75).

**Decision B.** Prompted-Qwen judging is permanently retired from the main
experiment. No further prompt variants; no lowered thresholds; neither holdout
opened; the bank stays unlaunched.

## Track 2 — the stress-test negative was a solver failure

Full record: `NONTRANSITIVITY_AUDIT.md`, artifacts in
`results/iclr2027_table1_v2/nontransitivity_audit/`.

`V_{k,beta}` is concave in `pi`, so `rho*`, the exact global Nash point and the
exact proximal Nash point are convex programs. Solved directly in float64 with a
tangent-plane certificate (gaps 6.6e-9 to 3.1e-7):

* **every v1 instance is inside Assumption 1** — `rho* > 0` on all 25, and the
  uniform reference is never the max-min optimum. But the margin shrinks 6.5x
  across the sweep, 0.111 → 0.017;
* the **exact** solutions reach 88–94 % of `rho*` at every alpha;
* the **deployed** alternating solver reaches −0.185 at alpha = 1 with a
  fixed-point residual of **1.000** and a weight norm of 608, sitting TV = 0.64
  from the exact proximal policy. At alpha = 0 the same code converges (residual
  6.3e-8) and lands on the exact policy to TV = 0.0000.

So the implementation is right and the *iteration* diverges once raw
`lambda = 1/s` explodes. The comparison the negative rested on is void as well:
fixed-reference Nash has a fixed-point residual of exactly 0 at every alpha
because its Eq. (21) map is constant, so a diverging iteration was being compared
against one that cannot diverge.

**"Roughly two thirds is step size" is retracted** with the arithmetic in
`step_size_claim_audit.json`: the fraction is 2.2 %, 78.1 %, 82.8 %, 46.9 % and
23.8 % at alpha 0 → 1.

## Track 3 — SafeRLHF cannot carry a cyclicity claim, and does carry a conflict one

Full records: `SAFERLHF_SUPERVISION.md`, `GPM_VS_BT.md`.

The released comparison graph is a **near-perfect matching**: 73 906 distinct
edges over 143 668 responses, only 1 369 of degree ≥ 2, and **zero triangles**
anywhere — verified independently under two definitions of response identity. The
directly observable three-cycle count is 0 by construction of the annotation
protocol, so no nontransitivity claim can be evidenced from this source, and the
pre-registered "observable nontransitive subset" test cannot be evaluated at all.

What SafeRLHF does carry is a large, genuine, two-dimensional conflict:
**24.33 % of held-out rows prefer different responses on helpfulness and on
harmlessness**, from independent human judgements on the same pair.

Matched-budget GPM vs BT (126.5M vs 126.4M parameters, 419 s vs 420 s): both beat
a constant predictor decisively; GPM's advantage over BT is −0.0039 nats on
helpfulness [−0.0073, −0.0005] and nothing on harmlessness, with no accuracy gain
on either. That null is what a matching graph predicts, and it is **not** evidence
that human preferences are transitive.
