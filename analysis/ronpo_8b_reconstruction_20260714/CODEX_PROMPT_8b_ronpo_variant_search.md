# CODEX PROMPT — Qwen3-8B: search for a RONPO variant that actually beats base on worst-objective (overnight, deadline 09:00 KST)

## 0. Goal, resources, deadline

**Goal.** In the `p1_8b_fair_demo_20260715` evaluation, EVERY trained method — RONPO included — loses to
`base` on the worst-objective panel (base is the 0.5 self-anchor; all trained models sit at 0.39–0.40).
RONPO is mid-pack among trained methods, not uniquely bad. The real phenomenon is an **alignment tax at
8B**: preference training on a very strong Qwen3-8B base erodes the worst objective. Your job: design,
train, and evaluate **theoretically-grounded RONPO variants** that PRESERVE base's worst objective while
nudging it up, so that at least one variant reaches or beats base (and the trained baselines) on the
worst-objective metric of Table~\ref{tab:qwen3-robust-validation} (main_v3.tex "Table 4"). Iterate:
discard losers, keep winners, upload the best to a public HF repo, and log everything to an md file.

**Resources / deadline.** 4 authorized B200 GPUs, all idle (verify read-only before every launch; never
touch another user's process). Hard deadline **09:00 KST** (from ~21:30 KST that is ~11.5 h). Reserve the
last ~1 h for final eval + HF upload + report. An 8B 900-step train ≈ 2–2.5 h/GPU; budget ~3 rounds of
4 concurrent variants with selection between rounds.

**Do NOT retrain baselines.** SPPO-avg, INPO-avg, DPO, HT-MNPO(safety/helpfulness) and their eval outputs
already exist under `results/p1_8b_fair_demo_20260715/`. Reuse them verbatim as the comparison set; spend
all compute on RONPO variants.

## 1. Hard prohibitions (violating any invalidates the run)

- Do **not** decode, rescore, tune on, or select on the spent 604-prompt sealed split
  (`results/p1_sealed_reward_seed42_20260714/`). It is spent.
- Do **not** choose the evaluator, objective set, seed, checkpoint, or metric *after* seeing which makes a
  variant win. Lock the evaluator + selection metric + primary hypothesis on a validity/power diagnostic
  and hash them **before** computing any variant ranking. No HARKing, no p-hacking, no cherry-picking.
- Do **not** fabricate numbers. Every reported number regenerates from JSON/CSV by a committed script.
- Report **all** variants tried, including discarded losers, with their hyperparameters and why they lost.
- If NO variant reaches base on worst-objective, the honest deliverable is that null + a scoped statement
  — do not manufacture a win. A variant that merely matches base's worst objective while improving the
  average is already a real, reportable result.

## 2. First, fix the judge so the confirmatory metric is valid

The `p1_8b_fair_demo_20260715` confirmatory primary was voided because 10 of 14,336 gpt-oss verdicts used
a `security` schema key where the frozen parser expected `safety`
(`results/p1_8b_fair_demo_20260715/fresh_test/judge/schema_alias_sensitivity_audit.json`). Before any new
ranking:
1. Patch the judge parser to accept the `security`↔`safety` alias (and any other observed alias) as the
   safety head, re-run ONLY the verdict-parsing/aggregation step over the EXISTING stored raw verdicts (no
   re-decode, no re-judge), and produce a corrected, now-valid confirmatory panel table for the existing 7
   models + base. Commit the patch + a before/after diff of the 10 affected verdicts.
2. This corrected panel is the frozen evaluator for the variant search. Hash it (`evaluator_lock.json`).

## 3. Diagnosis you are starting from

The worst-objective erosion has five compounding causes; each variant below targets one or more:
- **(a) Alignment tax vs a strong base.** Base is the 0.5 worst-objective anchor; all trained methods drop
  to ~0.40. Gentleness (trust-region to base) is the dominant lever.
- **(b) Deployed κ=0.05 is too soft.** Toy κ-sweep (`toy/os_ronpo_kappa_sweep.py`): the regularized-Nash
  hard worst-floor rises toward V* only as κ→0; at κ=0.05 it is BELOW top-mass and far below V*. RONPO's
  worst-objective advantage materializes only at κ≲0.01. Deployed κ makes RONPO behave like averaging.
- **(c) No worst-objective-to-base floor.** Nothing stops the worst objective drifting below base over 900
  steps; there is no worst-objective early stopping / checkpoint selection.
- **(d) top-mass instability / full-exp over-softness** at 8B; the knobs eta/K/tau/anchor were never swept
  for 8B.
- **(e) Objective starvation under top-mass** (one objective per update); full coverage needs OS-RONPO.

## 4. RONPO variants to train (theoretically grounded; prioritized)

All use the trainer `mnpo_scripts/mnpo_trainer.py` (`loss_type=ronpo`), config knobs in
`mnpo_scripts/mnpo_config.py` (`ronpo_alpha, ronpo_tau, reference_anchor_weight, preference_sft_weight,
eta`). Adversary temperature κ enters through the precomputed σ(k,a) built by
`mnpo_scripts/build_multi_objective_dataset.py` (its `kappa`); OS/full-exp/top-mass targets can be derived
CPU-only from stored per-objective scores via `mnpo_scripts/build_os_ronpo_targets.py` (already written,
validated to reproduce the stored `ronpo_objective_gap` to 1e-16). Rebuild σ/targets at the κ you need.

Round 1 (4 GPUs, the four highest-EV bets):

1. **RONPO-gentle-anchored + worst-objective checkpoint selection** *(highest EV — directly attacks the
   alignment tax).* full-expectation target, `reference_anchor_weight` 0.3–0.5, `ronpo_tau` 0.1, lr 5e-8,
   κ=0.02, save every 100 steps, and SELECT the checkpoint that maximizes validation worst-objective (not
   the last). Theory: τ·KL(π‖μ) + strong reference anchor keep the policy in a trust region around base,
   so the regularized Nash interpolates toward base as τ/anchor grow; early stopping prevents
   over-optimization. This is the variant most likely to reach base's worst objective.
2. **RONPO-κ-annealed** *(toy-validated worst-floor lever).* full-expectation target with κ annealed
   0.05→0.007 across steps (rebuild σ at a schedule, or piecewise datasets at κ∈{0.05,0.02,0.01,0.007}),
   moderate anchor 0.1. Theory: soft-min → hard worst-case V* as κ→0 (paper Prop. soft-min); γ=min(τ,κ)>0
   preserves last-iterate convergence. Sharpens the adversary onto the true worst objective.
3. **OS-RONPO (objective-stratified)** *(coverage + variance reduction).* target `Σ_k ω_k ẑ_{k,a_k}`,
   a_k~q(·|k), κ=0.02, anchor 0.1, via `build_os_ronpo_targets.py`. Theory: verified unbiased stratified
   estimator of the full-exp target (Lemma 1 / Prop 1–3), guarantees every objective enters each update.
4. **RONPO-CVaR/DRO-sharpened** *(robust-alignment inspired: MaxMin-RLHF / group-DRO).* replace the
   entropic soft-min over the objective marginal ω(k) with a CVaR_α (α≈0.3) worst-quantile weighting so
   mass concentrates on the worst objective(s); keep the response-level adversary as-is. Theory: CVaR is a
   coherent DRO risk on the objective distribution — a principled hard-floor surrogate distinct from κ.

Round 2 (pick from these based on Round-1 winners): the best Round-1 recipe × {stronger anchor 0.5, lr
2.5e-8, κ=0.005}; OS-RONPO × κ-anneal combo; gentle-anchored × CVaR combo; a longer/shorter-step variant
if worst-objective checkpoint selection suggests it. Round 3: refine the single best recipe (seed 42 only;
if a variant wins, optionally add seed 43 to check stability before upload).

Every variant: seed 42, same data pool, same 900-step budget (unless a variant's rationale is fewer steps
+ selection), symmetric with the frozen baselines. Log W&B run id per variant.

## 5. Evaluation — identical to Table 4 / fair_demo, pre-registered and locked

For each trained variant, run the SAME eval pipeline that produced `p1_8b_fair_demo_20260715`:
- **Stability gate (S3)** on the fresh prompt-disjoint 1,024-prompt UltraChat split (fail-closed; a
  variant that fails is discarded and logged as such).
- **Primary:** corrected gpt-oss panel **worst-objective win rate vs base** (the fixed §2 evaluator), with
  the pre-registered 2,000-resample bootstrap CIs. A variant "beats base" only if its worst-objective
  point estimate ≥ base AND the paired interval vs base excludes zero on the correct side.
- **Table-4 view:** ArmoRM helpfulness/safety/conciseness per-prompt min–max normalized Avg/Worst/WR_B/
  wWR_B, so the winner drops directly into `tab:qwen3-robust-validation`.
- **Independent RM secondary:** Skywork-V2 + Athene, worst-objective vs base.
- Do the ArmoRM/independent-RM scoring on validation for model SELECTION; open the confirmatory panel view
  only after the variant set and selection metric are locked (§2 hash). Never select on the primary.

## 6. Selection loop + HF upload

```
locked = frozen evaluator + selection metric (§2)
best = base            # incumbent to beat on worst-objective
for round in 1..3 (stop at 08:00 KST):
    launch up to 4 variants concurrently on the 4 B200s
    for each finished variant v:
        if v fails S3: log DISCARD(reason=stability); continue
        score v on validation (ArmoRM worst-obj + panel worst-obj vs base + indep RM)
        if v worst-obj > best worst-obj by a CI-separated margin:
            best = v; log KEEP
        else: log DISCARD(reason=numbers)
    pick next-round recipes from the round's winners
if best != base:
    upload best (weights, config, tokenizer, revision) to a PUBLIC HF repo under `promotion/`
      e.g. promotion/ronpo-qwen3-8b-worstobj-<recipe>-s42 ; verify AutoConfig reload + weight hash
    regenerate Table 4 numbers for `best` from JSON/CSV and update ronpo_aaai/main_v3.tex
      (replace/添加 only the RONPO rows; leave baseline rows untouched), rebuild main_v3.pdf clean
else:
    write the honest powered null; do not upload; leave main_v3.tex Table 4 unchanged
```

Upload ONLY a variant that beats or clearly matches base on worst-objective under the locked metric.

## 7. Required log file — `results/p1_8b_ronpo_variant_search_20260715/EXPERIMENT_LOG.md`

Maintain a running markdown log (update after every variant), containing:
- One row per variant: recipe name, full hyperparameters (estimator, κ/anneal schedule, anchor,
  preference_sft, tau, eta, lr, steps, checkpoint-selection rule), W&B run id, S3 pass/fail,
  validation worst-obj (ArmoRM) + panel worst-obj vs base + indep-RM, VERDICT ∈ {KEEP, DISCARD:stability,
  DISCARD:numbers}, one-line why.
- The theory note per variant (which of causes (a)–(e) it targets and the mechanism).
- Final section: the selected model (or the null), HF URL(s) + reload verification, main_v3.tex/pdf status,
  total GPU-hours, and `spent_sealed_split_touched=false` attestation.
- Also emit `COMPLETION_AUDIT.md` (SHA-256 of final main_v3.tex/pdf, evaluator_lock hash, W&B ids) and
  `summary.json` (machine-readable variant table).

## 8. Integrity guardrails (recap)

Spent sealed split never touched; evaluator + metric locked and hashed before ranking; baselines frozen
and symmetric; all variants (incl. losers) reported; numbers only from JSON/CSV; honest null allowed and
preferred over a manufactured win. Sample GPUs read-only before each launch; use only the 4 authorized
B200s; never modify another user's process.

## 9. One-paragraph brief (paste at top of the codex run)

> At Qwen3-8B, every trained method (RONPO included, mid-pack) loses to a strong base on the
> worst-objective panel — an alignment tax, not a RONPO-specific defect. First patch the gpt-oss judge
> `security`↔`safety` schema alias and re-aggregate the existing verdicts so the confirmatory panel is
> valid again, and hash it as the frozen evaluator. Then, keeping the existing baselines frozen, train
> theoretically-grounded RONPO variants that protect the worst objective — a gentle strongly-reference-
> anchored RONPO with worst-objective checkpoint selection (attacks the alignment tax), a κ-annealed RONPO
> (soft-min→V\* hard worst-case), objective-stratified OS-RONPO (unbiased full-coverage estimator), and a
> CVaR/DRO-sharpened adversary (robust-alignment) — four per round on the 4 idle B200s, seed 42, 900-step
> symmetric budget. Evaluate each with the exact Table-4 / fair_demo pipeline (ArmoRM per-objective worst,
> corrected panel worst-vs-base, independent RM), selecting on validation only; keep a variant only if its
> worst objective reaches/beats base with a CI-separated margin, discard and log the rest, iterate until
> 08:00 KST. If a winner emerges, upload it to a public `promotion/` HF repo (verify reload) and update
> only the RONPO rows of Table 4 in main_v3.tex with a clean rebuild; if none does, report the honest
> powered null and change nothing. Log every variant's hyperparameters, metrics, and KEEP/DISCARD verdict
> to `results/p1_8b_ronpo_variant_search_20260715/EXPERIMENT_LOG.md`. Never touch the spent sealed split;
> numbers only from JSON/CSV.
