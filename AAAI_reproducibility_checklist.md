# AAAI-27 Reproducibility Checklist — RONPO (draft to finalize before submission)

AAAI requires every author to complete this. Below is a pre-filled draft based on the
current paper. Items marked **[FIX]** must be true before you submit; items marked
**[AFTER PHASE 1]** depend on the experiments you still need to run. Answer honestly —
overclaiming here is worse than a "no."

## 1. General

- This paper makes theoretical contributions. **Yes.**
  - All assumptions and restrictions are stated clearly and formally. **Yes** (Lemma 1,
    Prop 1, Lemma 2 conditions; Limitations "Scope of the guarantee").
  - All novel claims are stated formally. **Yes.**
  - Proofs of all novel claims are included. **Yes** (Appendix "Deferred Proofs"; note in
    the paper that the theory is standard strongly-monotone-saddle machinery, not new math).
  - Proof sketches or intuitions are given for complex/informal results. **Yes.**
  - Appropriate citations to prior work are given. **Yes** (MaxMin-RLHF, group-robust PO,
    KL-DRO, SPPO/INPO/MNPO, reward soups, directional alignment).
  - The paper clearly delineates statements that are proved vs. conjectured. **Yes**
    (theory is exact for tabular dynamics; deployed loss is a plug-in surrogate — stated).

- This paper relies on datasets / makes experimental claims. **Yes.**

## 2. Data

- All datasets used are publicly available or described in detail. **[FIX]** UltraFeedback
  (public). **[AFTER PHASE 1]** add BeaverTails / XSTest / WildJailbreak for the
  conflicting-objective study; cite each.
- New datasets are described and will be released. **N/A** (no new dataset), unless you
  release the response pools + judge scores as an artifact (recommended — see §4).
- Preprocessing / normalization steps are described. **Yes** (per-prompt-per-objective
  min–max normalization over the fixed comparison set; described in appendix).

## 3. Models / reward judges

- Base policy identified with version. **Yes** — `Qwen2.5-1.5B-Instruct`. **[AFTER PHASE 1]**
  if you re-run on `Qwen3-1.7B/4B`, update.
- All reward models / judges identified with exact versions. **[FIX]** list every judge:
  `Skywork-Reward-V2-Llama-3.1-8B`, `Qwen3Guard-Gen-4B` (safety, if used), deterministic
  brevity band; external judge `gpt-5.5-2026-04-23` with exact date.
- External-judge protocol documented (prompt, order randomization, tie handling).
  **Yes** (Appendix "External-Judge Protocol"; ties = 0.5, order randomized).

## 4. Code & experimental reproducibility

- Code will be released, or enough detail is given to reproduce. **[FIX]** decide: release
  a code+data archive (allowed as supplementary, due 3 days after paper) — strongly
  recommended, and note that AAAI's AI-Alignment track explicitly favors released tools.
- Training details (optimizer, LR, schedule, steps, batch, hardware) are reported.
  **[FIX]** currently in appendix for the runs shown; ensure the **checkpoint-selection
  rule** (held-out worst-objective floor, matched step budget) is stated — this is what
  makes the atom-vs-weight comparison fair.
- Number of runs / seeds reported. **[AFTER PHASE 1]** currently **single seed** — the
  paper says so in Limitations. For any *headline* claim you keep, report **≥3 seeds**.
- Error bars / significance defined. **Yes for prompt-level** (paired bootstrap 95% CI);
  **[AFTER PHASE 1]** add **seed-level** CIs for headline numbers, or do not claim
  significance across seeds (the paper currently does not — keep it that way if seeds stay 1).
- Hyperparameter search / selection described. **[FIX]** report `τ, κ, η`, pool `A(x)`
  construction, and how they were chosen; report the `κ` sweep (now in appendix figure).

## 5. Compute

- Compute resources described (hardware, wall-clock). **[FIX]** state GPUs
  (2×A100 + shared H200) and approximate GPU-hours per run.

## 6. Honesty items (do not skip)

- Claims in the abstract/intro are supported by the results. **[FIX after Phase 1]** the
  abstract currently claims model-scale worst-objective gains on *correlated* judges; if
  the conflicting-objective run doesn't land, soften the abstract to match (the current
  draft already frames model-scale as "heterogeneous-reward robustness," not "opposed
  objectives" — keep that alignment).
- Limitations are discussed. **Yes** (dedicated section: transitive plug-in oracle,
  correlated judges, single seed, tabular-only guarantee, parity-with-base on hardest subset).
- Potential negative societal impacts considered. **[FIX]** add one line (worst-case
  robustness can entrench a mis-specified safety judge; mitigations = judge auditing).
- **No fabricated or estimated numbers.** **Yes** — every reported value is a real
  measurement; unfinished experiments are left as `% TODO`, not filled with guesses.
  Keep it that way.
