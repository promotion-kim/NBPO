# STATUS — ICLR-2027 Table-1 rebuild

Updated 2026-09-07, end of session 1. Branch `exp/iclr27-table1-v2`.

Nothing below is a projection. Runs that are still going are named as such, with
their log path, and no number is quoted from them.

## Gate summary

| gate | state |
|---|---|
| baseline suite unchanged | **pass** — 219 → 305 passed, 2 documented skips, 0 failures |
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
| judge audit: **swap consistency** | **FAIL** — 0.59–0.70 against a 0.85 gate. Diagnosed, not lowered. See below. |
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

| run | where | log |
|---|---|---|
| rubric diagnostic (v1@16 tokens, v2@64 tokens), Qwen3-32B TP=2 | pod `nbpo-judge`, GPUs 0–1 | `/work/iclr27_table1_v2/logs/diag_judge.log` |
| 70B cross-judge of the same pairs (chained behind it) | pod `nbpo-judge`, GPUs 0–1 | `/work/iclr27_table1_v2/logs/diag_70b.log` |

Completed on the pod: both response pools (02:46–02:49 PDT, one GPU) and the
50-prompt training judgment bank (02:51–02:57 PDT, Qwen3-32B TP=2) —
8800 rows at `/work/iclr27_table1_v2/smoke/verdicts.jsonl`.

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
  same 16-token budget gives **86.5 % unparseable** — v1's phrasing makes the
  judge write a preamble, and the published campaigns used a 512-token budget
  with it. v2 is 0 % at 16 tokens. So v2 is strictly better on the axis it was
  meant to fix, and the consistency number is not a v2 artifact.

The live hypothesis is the **pool**, not the judge: eight samples from one 8B
model on one prompt are frequently near-equal, and a judge asked which is
*materially* better on a narrow criterion has little to latch onto. The tie
rates say the same thing — 0.40 to 0.66 swap-averaged, and 580 of 1100 honesty
pairs are ties in both orders. Under that reading the low consistency is the
measurement being honest about indifference, and swap averaging is doing exactly
the job it exists for.

That hypothesis is being tested directly: the same pairs, same rubric, same
decoding, judged by **Llama-3.3-70B** (`/work/models/xj_judges/llama70`, TP=2,
`--judge-role monitoring` so it can never be mistaken for the training bank). If
a far stronger judge lands near 0.65 the cause is the pool; if it reaches 0.85
the cause is Qwen3-32B. **That run is in flight and no conclusion is drawn from
it here.**

Whichever way it resolves, the full labelling run stays blocked until the gate
is met or the protocol's threshold is revised deliberately and on the record.

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

1. finish smoke judging → tensors → solve → pair targets → short 8B fit → decode
   → independent evaluation (closes execution-order step 9);
2. 200-prompt judge audit against the hard gates (step 10);
3. full train/validation pool and judgment bank (step 11);
4. beta/R diagnostics, then the sequential eta/LR/step pilot on tuning seed 11
   (steps 12–13);
5. regression-realization gate, then freeze and launch seeds 42/43/44.

RACO is **not** started: `references/chen_raco_icml_2026.pdf` is present but no
official implementation has been located, and the brief forbids inventing one.
It stays marked blocked until a faithful source is pinned.
