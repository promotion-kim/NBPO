# CODEX PROMPT — Qwen3-8B: give RONPO its strongest *legitimate* shot at a demonstrated worst-objective advantage

## 0. What this task is, and is NOT

**Goal:** the current Qwen3-8B section reports no RONPO advantage. Diagnose *every* real reason the
signal is being suppressed at 8B, fix all of them, and run the strongest **honest, pre-registered**
experiment that gives RONPO a fair chance to show a genuine worst-objective robustness advantage over
base and all baselines. If RONPO truly has an 8B advantage, this design surfaces it credibly. If it does
not, this design tells us that cleanly and we scope the claim.

**This is NOT** a task to manufacture a positive result. Hard prohibitions (violating any one invalidates
the whole run):
- Do **not** decode, rescore, tune on, or select on the spent 604-prompt sealed split
  (`results/p1_sealed_reward_seed42_20260714/`). It is spent.
- Do **not** choose the evaluator, objective set, seed, checkpoint, or metric *after* seeing which one
  makes RONPO win. No HARKing, no p-hacking, no cherry-picking. The evaluator and the selection metric
  are locked on a validity+power diagnostic **before** any method ranking is computed, and hashed.
- Do **not** fabricate or hand-edit numbers. Every reported number is regenerated from JSON/CSV by a
  script committed in the run directory.
- Every baseline (SimPO, IPO, SPPO-avg, INPO-avg, DPO, HT-MNPO) gets a **symmetric** tuning budget. A
  RONPO win obtained by tuning RONPO harder than the baselines is not a win; reviewers will see it.
- Report **all** pre-registered outcomes, including the ones where RONPO loses. Keep all official
  aggregations (e.g. all four IFEval views), not a single favorable one.

If, after this, RONPO does not win the pre-registered primary on a held-out test with a demonstrably
powered evaluator, the deliverable is the honest powered result + a scoped claim, not a positive spin.

## 1. Diagnosis you are starting from (verified from data/code)

Root cause splits in two:

**(A) The sealed evaluation had ~zero statistical power (dominant on the headline table).**
- ArmoRM helpfulness/safety/conciseness raw means span only 0.0027–0.0045 across all 10 models
  *including base* (`ronpo_aaai/qwen3_power_diagnostic_macros.tex`,
  `results/p1_sealed_reward_seed42_20260714/results/per_objective_scores.csv`). No method moved these
  heads measurably at 8B.
- The reported worst-objective order comes from **per-prompt min–max normalization amplifying sub-noise
  variation** into a 0.216–0.235 range. It is amplified noise, not a ranking.
- Independent RMs confirm the null, not "wrong evaluator": Skywork 0/45 pairs, Athene 3/45 pairs, both 0
  trained-vs-base (`results/p1_sealed_reward_seed42_20260714/power_diagnostic/REPORT.md`). Qwen3-8B base
  is already strong; KL-regularized preference nudges land inside evaluator noise (policies do move —
  ~77% of gens differ, word Jaccard 0.56 — but below reward resolution).

**(B) At 8B the RONPO estimators were untuned and one regressed (the genuine, narrow negative).**
- top-mass (`ronpo_k_only`) is unstable at 8B: it targets a hard 1.0 on the worst atom → repetition
  collapse (max identical-word run 1607) → fails the stability gate
  (`results/p1_8b_retrain_stronger_20260715/validation/results/REPORT.md`).
- full-expect converges but conciseness regressed (−0.00890 [−0.01619, −0.00152]) and base led normalized
  worst-objective (base 0.2033 > RONPO full-exp 0.1857).
- Hyperparameters were a naive 1.5B→8B scale (`results/p1_8b_retrain_stronger_20260715/config.json`:
  lr 1e-7→5e-7, steps 900→1800, eta 0.0075, tau 0.05, alpha 0.5, anchors 0.05/0.005). The knobs that
  actually govern RONPO's worst-objective behavior — adversary sharpness `eta`, mass-truncation `K`,
  logistic temp `tau`, `reference_anchor_weight` — were **never swept for 8B**.

So the headline table is an underpowered-evaluator artifact and the retrain negative reflects untuned 8B
optimization + a known top-mass instability. Neither is evidence the method is fundamentally wrong at 8B.
Both are fixable on the legitimate axes below.

## 2. Environment / repo facts

- Trainer: `mnpo_scripts/mnpo_trainer.py`, config `mnpo_scripts/mnpo_config.py`
  (`reference_anchor_weight`, `preference_sft_weight`, tau/eta/alpha live here and in run config.json).
- RONPO estimator variants: `ronpo_full_expect` (Rao–Blackwellized, `eq:rb-target`) and `ronpo_k_only`
  (top-mass, `eq:ronpo-loss-empirical`). Names `k_only`/`full_expect` do NOT appear in the .tex; paper
  terms are "top-mass" vs "full-expectation".
- Paper working file: `ronpo_aaai/main_v2.tex`. Numeric fragments are auto-generated:
  `qwen3_stronger_validation_table.tex`, `qwen3_power_diagnostic_*.tex` by scripts under
  `analysis/ronpo_8b_reconstruction_20260714/` (`build_stronger_validation_table.py`,
  `build_qwen3_power_diagnostic_table.py`, `build_ronpo_8b_tables_figure.py`).
- Stability gate: `scripts/revision/flagship/stability_gate_corrected.py` (fail-closed; counts non-empty
  `<think>…</think>` leak spans + max identical-word run; keep it unchanged and reward-blind).
- Hardware: 4 authorized B200 GPUs. Sample GPUs read-only before launch; never stop or modify another
  user's process. `enable_thinking=false` decode; seed 42 unless a seed sweep is pre-registered.
- New run directory for everything below: `results/p1_8b_fair_demo_<UTC-date>/`. `Date.now()` etc. are
  fine in shell; stamp timestamps from the shell, not fabricated.

## 3. Workstream 1 — build an evaluator with demonstrated power AND validity (pre-register, then lock)

The single biggest reason RONPO "loses" is that nothing separates at all. Fix the measuring instrument
first, and prove it can measure before you use it to rank.

1. **Assemble candidate objective sets** that are (i) genuinely conflicting and (ii) non-saturated at 8B.
   Do not reuse the ArmoRM heads blindly — first test them. For each candidate reward model / head:
   - **Resolution test:** it must separate base from a deliberately weakened policy (e.g. a
     length-degenerate or off-objective checkpoint) with a paired 95% bootstrap interval excluding zero.
     A head that cannot separate a known-worse policy from base is saturated → **discard it**.
   - **Conflict test:** the chosen objectives must have low/negative cross-objective Spearman on the
     candidate pool (target ≤ 0, report the matrix) AND a high top-1 objective mismatch. Conflict is the
     precondition for RONPO's worst-objective floor to matter.
   - Prefer reward models known to be sensitive and mutually disagreeing at 8B (candidates:
     Skywork-Reward-V2, Athene-RM-8B, ArmoRM heads that pass resolution, plus a length/verbosity signal).
2. **Kill the noise amplifier.** Do not rank on per-prompt min–max normalization. Pre-register a
   normalization that does not blow up sub-noise variation: paired raw-delta tests as the primary, plus a
   base-anchored z-score (or rank-based) worst-objective as a secondary. Report the raw per-objective
   deltas explicitly so a null is visible as a null.
3. **Power the sample.** Do a real power analysis: from the measured per-prompt SD, compute the prompt
   count n needed to detect a pre-registered target effect (e.g. a worst-objective Δ of the size seen at
   1.5B) at 80% power. Scale the prompt pool to meet it. If the accessible pool cannot reach the required
   n, say so and report the minimum detectable effect instead of implying power you don't have.
4. **Add a reward-model-independent judge panel** as the primary robustness signal, since local RMs
   overlap training objectives. Use ≥2 strong LLM judges, position-swapped, with inter-judge agreement
   and per-objective win-rates vs base. This is the signal reviewers will trust most.
5. **Pre-register and hash** the objective set, evaluator identities+revisions, resolution/conflict test
   outcomes, primary metric, primary hypothesis, n, and the full analysis plan **before** computing any
   method ranking. Write `PREREG.md` + `evaluator_lock.json` (SHA-256) into the run dir.

## 4. Workstream 2 — tune 8B optimization on a validation split (symmetric across methods)

Give RONPO's estimators a fair chance to actually move the worst objective and to stay stable — and give
the baselines the identical budget so any win is real.

1. **Sweep the knobs that govern RONPO** on the non-sealed validation split (NEVER the sealed split):
   - `eta` (adversary step / worst-objective sharpness), `K` (mass-truncation / adversary support),
     `tau` (logistic temperature), `learning_rate`, `reference_anchor_weight`,
     `preference_sft_weight`, `optimizer_steps`, `warmup_ratio`.
   - For **top-mass stability**: the hard-1.0-on-worst-atom target is the known failure mode. Legitimate
     fixes to try: anneal adversary sharpness (start softer `eta`/larger `K`, sharpen over training),
     stronger reference/SFT anchor for top-mass specifically, lower peak lr, longer warmup. Success
     criterion: passes the unchanged stability gate (no repetition collapse) AND moves the worst
     objective. Do not weaken/rig the gate to pass a degenerate model.
2. **Selection is on validation, by the pre-registered metric only.** Pick the best config per method on
   validation; never peek at the test. Log the full sweep grid + per-config validation metric to
   `sweep/` and W&B.
3. **Symmetric baseline tuning.** Run an equally serious sweep for SimPO/IPO/SPPO-avg/INPO-avg/DPO/
   HT-MNPO (their own principled knobs, comparable compute budget). Record the budget so it is auditable.
   A credible RONPO win requires the baselines to be at *their* best, not RONPO at its best vs baselines
   at defaults.
4. **Report both estimators.** Carry forward whichever RONPO variant wins validation under the
   pre-registered rule, but report both top-mass and full-expectation in the appendix (the avg-vs-worst
   trade-off is itself a result).

## 5. Workstream 3 — confirm on a held-out test, then report honestly

1. Take the validation-selected configs to a **fresh, prompt-disjoint sealed split**: write the manifest,
   hash it, decode once, score once with the locked evaluator. Do not reuse or reopen the spent split.
2. Apply the pre-registered analysis exactly as written. Regenerate every paper number from JSON/CSV via
   a committed script; produce updated `qwen3_*` .tex fragments for `main_v2.tex`.
3. **Decision rule (write the outcome that is true):**
   - If RONPO wins the pre-registered primary vs base and beats baselines on worst-objective with the
     powered evaluator → report the strong 8B claim, with power analysis, symmetric-tuning evidence, and
     the independent judge panel as support.
   - If RONPO wins on some objectives/judges but not others → report the mixed result faithfully (as the
     IFEval paragraph already does), no single-metric spin.
   - If RONPO still does not separate → report the **powered null** honestly and scope the 8B claim to
     theory + synthetic + 1.5B. A credible powered null is a *publishable, defensible* outcome and far
     safer than an unreproducible positive.

## 6. Deliverables (into `results/p1_8b_fair_demo_<UTC-date>/`)

- `PREREG.md`, `evaluator_lock.json` (+SHA), power analysis, resolution/conflict test results.
- `sweep/` full validation grid (RONPO + all baselines) + W&B run IDs.
- Fresh-sealed manifest + hash + scores; `REPORT.md` with the honest decision-rule outcome.
- Regenerated `ronpo_aaai/qwen3_*.tex` fragments + a clean TinyTeX build of `main_v2.tex`
  (0 fatal / 0 unresolved refs+cites / 0 overfull), with SHA-256 of the final .tex and .pdf.
- `COMPLETION_AUDIT.md` + `fix_log.md` recording exactly what was tuned, what passed/failed, and
  `spent_sealed_split_touched=false` throughout.

## 7. One-paragraph brief (paste at top of the codex run)

> Diagnose and fix why RONPO shows no advantage on the Qwen3-8B section of `ronpo_aaai/main_v2.tex`. The
> two root causes are (A) an underpowered/saturated evaluator (ArmoRM head spread ≤0.0045; independent
> RMs separate 0 trained models from base) and (B) untuned 8B optimization with an unstable top-mass
> estimator (repetition collapse; full-expectation conciseness regression). Give RONPO its strongest
> *legitimate* chance: build and pre-register an evaluator proven to have resolution and genuine objective
> conflict, run a real power analysis, tune all methods symmetrically on a validation split (stabilize
> top-mass; sweep eta/K/tau/lr/anchors), then confirm the validation-selected configs once on a fresh,
> prompt-disjoint sealed split with an independent judge panel as the primary signal. Do NOT touch the
> spent sealed split, choose the evaluator/metric before seeing rankings and hash it, tune baselines as
> hard as RONPO, regenerate every number from JSON/CSV, and report all outcomes — including a powered null
> — honestly. If RONPO wins under this design the result is credible; if it does not, deliver the powered
> null and a scoped claim.
