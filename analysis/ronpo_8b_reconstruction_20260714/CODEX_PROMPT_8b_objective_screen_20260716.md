# CODEX PROMPT — Qwen3-8B objective resolution and conflict screen (inference-only; decide if RONPO can be shown at 8B)

## 0. Goal

Decide, before any RONPO training, whether an objective set exists at Qwen3-8B that can actually exercise
RONPO's worst-objective mechanism. RONPO needs three things at once: **resolution** (the objective can tell a
better policy from a worse one), **conflict** (objectives genuinely trade off, so averaging can sacrifice a
minority one), and **base headroom** (the base is beatable on the weakest objective). The 8B section failed
because the ArmoRM heads conflict but saturate, and the general reward models (Skywork, Athene, ArmoRM)
appear correlated at 8B. This run measures both, with no training and no upload, and outputs the best
qualifying objective triple or an honest "none qualifies."

**This is a measuring-instrument screen, not a RONPO run.** Objectives are ranked only by resolution,
conflict, and headroom, which are properties of the reward models and the base policy. Do not select the
objective set by which one would make RONPO win; that decision is made here, before RONPO is trained, on
criteria that do not reference RONPO's ranking.

## 1. Hard rules

- Inference only: decode and reward-score. No preference training, no HuggingFace upload, no Table edits.
- Never decode, score, or read the spent 604-prompt sealed split (`results/p1_sealed_reward_seed42_20260714/`).
- **Pre-register and hash** the candidate policy set, the objective menu, and the pass thresholds (resolution
  CI rule, conflict Spearman target, headroom rule) **before** computing any correlation or ranking. Write
  `PREREG.md` + `screen_lock.json` (+SHA).
- Regenerate every number from JSON via a committed script. Report the full matrices, including objectives
  that fail.
- Use available GPUs (4 B200, H200 `air3` if a read-only check shows them free, free odin2 GPUs; never
  another user's GPU). Minimal local storage: keep score JSONL, delete raw generation shards once merged.

## 2. Build the scoring pool

**Prompts.** Two subsets, hashed manifests, both disjoint from the spent split:
- `general`: the existing 647 held-out prompts (`data/gemma2_ufb_part2_test.jsonl`).
- `conflict_curated`: a subset selected or written to create genuine objective tension (prompts that
  simultaneously demand help, safety, and brevity, e.g. sensitive-topic questions that need a careful,
  complete, and short answer). Conflict may only appear on such prompts; document how they were chosen.

**Policies to score** (decode each with `scripts/revision/flagship/decode_vllm_non_thinking.py`, seed 42,
non-thinking template):
- `base`: Qwen3-8B.
- **Known-worse degradations (the resolution probes).** Each is a policy we know a priori is worse on one
  objective; a non-saturated objective must separate it from base:
  - `weak_small`: a much smaller model (e.g. Qwen2.5-1.5B-Instruct or Qwen3-0.6B) for general-quality
    resolution.
  - `verbose`: base decoded under an "answer at maximum length and detail" instruction (worse on
    conciseness/brevity).
  - `terse`: base decoded under an "answer in at most one short sentence" instruction (worse on
    helpfulness/completeness).
  - `less_aligned`: base decoded under a mild "be blunt and skip caveats" instruction, or an existing older
    checkpoint, for safety resolution. Keep it defensive; do not attempt to elicit genuinely harmful content.
- `trained_pool`: the existing stable trained 8B checkpoints (RONPO arms and baselines that pass the gate),
  to measure real cross-policy spread and whether any policy beats base (headroom).

## 3. Objective menu (score every policy's responses with each)

- General reward models: **Skywork-Reward-V2** (`on_policy_data_gen/rm_skywork.py`), **Athene**
  (`rm_athene.py`), **ArmoRM whole** (`rm_armo.py`). These are the Table-2 / Option-A objectives.
- **ArmoRM heads**: helpfulness, safety, conciseness (from `rm_armo.py`). The current 8B set.
- **Safety critic**: Qwen3Guard (`rm_qwen3guard.py`).
- **Brevity / verbosity** signal (`mnpo_scripts/score_brevity_and_collapse.py`).
- Optionally a verifiable-constraint or format signal if one is cheap to add (IFEval-style), since a strong
  base can still be weak on strict constraints.

Reuse the existing conflict tooling: `mnpo_scripts/build_conflict_objective_scores.py` and
`mnpo_scripts/analyze_conflict_gate.py`.

## 4. Pre-registered tests (lock thresholds before computing)

For each objective compute, on the general and conflict_curated prompts:

1. **Resolution.** Paired prompt-level bootstrap (2000 resamples, seed 42) of the score gap between `base`
   and the matched known-worse policy (verbose for conciseness/brevity, terse for helpfulness, less_aligned
   for safety, weak_small for the general models). **Pass if the 95% CI excludes 0.** Also report the minimum
   detectable effect. An objective that cannot separate a known-worse policy from base is **saturated ->
   discard**.
2. **Conflict.** Full pairwise cross-objective Spearman correlation matrix over the pooled per-prompt scores,
   plus the top-1 objective mismatch rate. A pair is conflicting if Spearman <= 0. Report the matrix.
3. **Headroom.** For each objective, is there any policy in `trained_pool` that beats `base` with a paired
   95% CI excluding 0? If no policy can exceed base on an objective, there is no worst-objective for RONPO to
   raise there; flag it.

## 5. Decision

- Enumerate candidate objective **triples** from the menu (and any conflict-tooling composites). A triple
  **qualifies** if: all three objectives pass resolution, every pair has Spearman <= 0 (genuine conflict),
  and at least the weakest objective has headroom (some policy beats base). Rank qualifying triples by the
  strength of their conflict (most negative mean pairwise Spearman) and resolution margin.
- **Outcome A:** a qualifying triple exists -> report it as the objective set for a subsequent RONPO 8B run,
  with its resolution, conflict, and headroom evidence. (If that triple is exactly Skywork/Athene/ArmoRM, the
  Table-2 scale-up path is alive; if it is a harder set, that is the path.)
- **Outcome B:** no triple qualifies -> report the honest negative: at 8B the available objectives cannot
  simultaneously resolve and conflict with base headroom, so the model-scale robustness claim stays scoped to
  1.5B and the synthetic games. This is a publishable, defensible result, not a failure.

## 6. Deliverables (into `results/p1_8b_objective_screen_20260716/`)

- `PREREG.md` + `screen_lock.json` (+SHA); both prompt manifests (+SHA).
- `generations/` (deleted after scoring), `scores/` per objective per policy.
- `RESOLUTION.md` (per-objective base-vs-degraded gaps + CIs + minimum detectable effect).
- `CONFLICT.md` (full Spearman matrix + top-1 mismatch, for general and conflict_curated).
- `HEADROOM.md` (which policies beat base per objective).
- `DECISION.md` with the qualifying triple(s) or the honest "none," and the recommended next step.
- The committed script that regenerates every number from the score JSONL.
- `COMPLETION_AUDIT.md`, ending `spent_sealed_split_touched=false`.

## 7. One-paragraph brief (paste at top of the codex run)

> Before training any more RONPO at Qwen3-8B, run an inference-only screen to decide whether an objective set
> exists that can exercise RONPO's worst-objective mechanism. RONPO needs resolution (the objective separates
> a better policy from a worse one), conflict (objectives trade off so averaging can sacrifice a minority
> one), and base headroom (the base is beatable on the weakest objective). Decode base, a small weak model,
> and per-objective known-worse degradations (verbose, terse, less-aligned), plus the existing stable trained
> 8B checkpoints, on the 647 held-out prompts and a conflict-curated subset. Score everything with Skywork,
> Athene, ArmoRM whole and its helpfulness/safety/conciseness heads, Qwen3Guard, and a brevity signal.
> Pre-register and hash the policy set, objective menu, and pass thresholds first. Then measure, per
> objective, resolution (paired bootstrap of base minus the matched known-worse policy, 95% CI must exclude
> zero, else the objective is saturated and discarded), conflict (full pairwise Spearman, conflicting if <=
> 0), and headroom (does any policy beat base). Output the best objective triple that passes all three, or an
> honest "none qualifies." Do no training, no upload, no Table edits, never touch the spent sealed split,
> regenerate every number from JSON, and use available GPUs without touching another user's job. If a triple
> qualifies it becomes the objective set for a later RONPO run; if none does, the model-scale claim stays
> scoped to 1.5B plus synthetic.
