# CODEX PROMPT — Qwen3-8B RONPO on the helpful-vs-harmless conflict

## 0. Why this is the right objective set

The 8B section failed because its objectives either conflict but saturate (ArmoRM heads) or resolve but do
not conflict (general reward models agree because the strong base scores high on all). The helpful-vs-harmless
tradeoff avoids both failure modes and is the canonical multi-objective alignment conflict:

- Helpfulness and harmlessness are intrinsically opposed on sensitive prompts: answering helpfully can raise
  harm, refusing raises safety but lowers helpfulness. Recent analysis measures a helpfulness-harmlessness
  Spearman near -0.57 among top reward models, far stronger conflict than the -0.10 of the ArmoRM heads.
- Safe-RLHF decouples a helpfulness reward model from a harmlessness cost model precisely because the two
  objectives conflict, and reports the "exaggerated safety" failure (harmless but unhelpful). That minority
  objective collapse is exactly what RONPO's worst-objective floor is meant to prevent.
- A strong Qwen3-8B base still has real headroom here: on borderline prompts it either over-refuses (helpful
  low) or under-refuses (harmless low). This is the regime where the base is beatable on the worst objective.
- This directly instantiates the paper's own motivating example ("the failure a safety or truthfulness
  critic is supposed to prevent," Introduction), now at the larger backbone.

Goal: find the most-conflicting helpful and harmless reward-model pair that also resolves at 8B, train RONPO
(objective-stratified lead arm) and the averaging baselines on it over a safety-relevant prompt distribution,
and test whether RONPO protects the worst objective while averaging methods sacrifice a minority one. If
RONPO leads, upload it and add the result to the paper. If not, report the honest result.

## 1. Hard rules

- Pre-register and hash the reward-model pair, prompt manifests, metric, and pass thresholds before any
  ranking. No HARKing: the objective pair is chosen in Phase 1 on conflict and resolution, not on which pair
  makes RONPO win.
- Never touch the spent 604-prompt sealed split (`results/p1_sealed_reward_seed42_20260714/`).
- Symmetric budget across methods. Keep the stability gate
  `scripts/revision/flagship/stability_gate_corrected.py` unchanged and reward-blind; harden training, not the
  gate.
- Regenerate every number from JSON. Report all outcomes, including where RONPO loses.
- Use available GPUs (4 B200, H200 `air3` if free after a read-only check, free odin2 GPUs; never another
  user's GPU). Minimal local storage; delete large artifacts after use or upload.

## 2. Phase 1 — pick the conflicting, resolving helpful-harmless pair (inference only)

**Prompt distribution (safety-relevant, where the conflict lives).** The current UltraFeedback prompts
(`data/gemma2_ufb_part2_test.jsonl`) are general and benign, so helpfulness and harmlessness do not conflict
on them; a good answer is both helpful and safe. Switch to a safety-relevant mix so that both failure modes
appear. Assemble and hash a pool from public datasets, with both over-refusal and genuinely-harmful cases:
- **Core:** `PKU-Alignment/PKU-SafeRLHF` prompts (dual-preference helpfulness and harmlessness, 19 harm
  categories, borderline by design). This is the primary training pool.
- **Over-refusal (surfaces helpful-low):** OR-Bench hard subset (`huggingface.co/bench-llms`) and XSTest
  (`allenai/xstest-response`), benign prompts that look unsafe, where a strong base over-refuses.
- **Harmful slice (surfaces harmless-low):** a portion of higher-severity `PKU-Alignment/BeaverTails` or
  PKU-SafeRLHF prompts, where a helpfulness-chasing policy under-refuses.
Mix so both over-refusal and under-refusal are present (do not use only clearly-harmful prompts). Keep a
held-out validation subset and a disjoint fresh-confirmation subset. Do not use the spent split. All are
public HuggingFace datasets and need downloading; record versions and hashes.

**Candidate reward models.**
- Helpfulness: Skywork-Reward-V2 (`on_policy_data_gen/rm_skywork.py`), Athene (`rm_athene.py`), ArmoRM
  helpfulness head (`rm_armo.py`). (Skywork-Reward-V2-Qwen3-8B matches the policy backbone if worth adding.)
- Harmlessness: start with the local signals, the ArmoRM safety head (`rm_armo.py`) and Qwen3Guard
  (`rm_qwen3guard.py`, unsafe probability as cost). If neither resolves at 8B, download and add a stronger,
  purpose-built harm signal (a Safe-RLHF / BeaverTails cost model, or a Llama-Guard-style moderation model)
  and write a matching scorer mirroring the existing `rm_*.py`.

**Scoring pool.** base Qwen3-8B; per-objective known-worse probes for the resolution test (`terse` and
`over-refusing` for helpfulness-low, `answer-anything` for harmlessness-low); a small weak model; and the
existing stable trained 8B checkpoints.

**Measure and lock.**
- Resolution: paired prompt bootstrap of base minus the matched known-worse policy, 95% CI must exclude 0,
  else the reward model is saturated and discarded.
- Conflict: pairwise Spearman between each helpfulness and each harmlessness reward model over the pooled
  per-prompt scores; pick the pair with the most negative Spearman that also passes resolution for both.
  Report the full matrix.
- Headroom: on the chosen pair, verify the base is beatable on the weaker objective (some policy exceeds base
  with a CI excluding 0).

Output `PAIR_DECISION.md` with the selected (helpful RM, harmless RM) pair and its conflict, resolution, and
headroom evidence. If no pair conflicts and resolves, stop and report that honest negative.

## 3. Phase 2 — train and evaluate on the chosen pair

Only if Phase 1 yields a conflicting, resolving pair.

1. Build the two-objective (optionally add a third axis such as brevity or honesty to keep K=3) RONPO dataset
   on the safety-relevant prompts: decode the Stage-1 pool, score with the chosen helpful and harmless reward
   models, build targets with `mnpo_scripts/build_os_ronpo_targets.py` (`target_os_k{K}`) and the averaged
   pairs with `mnpo_scripts/build_multi_objective_dataset.py`.
2. Train, symmetric budget, seed 42, mu = base as the KL anchor, collapse-hardened (reference anchor 0.3 to
   0.5, soft-to-hard kappa annealing, peak lr <= 2.5e-8, warmup >= 0.2, nonzero preference-SFT):
   - RONPO-OS (lead), plus full-expectation and stabilized top-mass.
   - Averaging baselines SPPO-avg, INPO-avg, and DPO (the fixed-average methods RONPO is meant to beat on the
     worst objective).
3. Select every checkpoint on the full validation set (never a 128-prompt subset); keep only stability-gate
   passers.

## 4. Metric, confirmation, decision

- Primary: worst objective as the minimum over the two (or three) objectives of the per-objective score vs
  base, with paired bootstrap CIs; report both the helpfulness and harmlessness deltas, the avg-vs-worst
  tradeoff, and win rate vs base. Add an independent judge (safety-aware) as the confirmatory signal.
- The key comparison to make explicit: do the averaging baselines raise the average while dropping one
  objective (the "exaggerated safety" or the opposite), and does RONPO hold both above a higher floor? That
  Pareto contrast is the result even if the absolute margin vs base is modest.
- Fresh confirmation once on the disjoint subset; never the spent split; no peek-and-continue.
- Decision: best RONPO arm has the highest worst objective among all methods -> upload to a public
  HuggingFace repo and add the result to the paper (a helpful-harmless robustness table or an addition to the
  8B section), scoped honestly. Leads baselines but not base -> report as best trained method, scope. Else
  honest null, no upload, no paper edit.

## 5. Deliverables (into `results/p1_8b_helpful_harmless_20260716/`)

- `PREREG.md` + `metric_lock.json` (+SHA); prompt manifests (+SHA); `PAIR_DECISION.md` with the conflict and
  resolution matrices.
- `sweep/` per-step validation gate results and metrics + W&B IDs.
- `fresh/model_summary.json` + per-objective CSVs; the committed aggregation and table script.
- `REPORT.md` with the decision, the helpfulness and harmlessness deltas, the avg-vs-worst Pareto, and the
  per-step stability profile.
- On a win: HF repo URL + commit; regenerated paper fragment; rebuilt `main_v3.pdf` (0 fatal, 0 unresolved
  refs and cites, 0 overfull); SHAs; deleted-files list.
- `COMPLETION_AUDIT.md` + `fix_log.md`, ending `spent_sealed_split_touched=false`.

## 6. One-paragraph brief (paste at top of the codex run)

> Run RONPO at Qwen3-8B on the canonical helpful-vs-harmless conflict, the setting the paper's introduction
> already motivates and where reward models genuinely disagree (helpfulness-harmlessness Spearman near -0.57),
> unlike the saturated ArmoRM heads or the correlated general reward models. Phase 1 (inference only): on a
> safety-relevant, borderline prompt distribution (PKU-SafeRLHF / BeaverTails, HH harmless-base, an
> over-refusal set like XSTest), score base, per-objective known-worse probes, a small weak model, and the
> stable trained 8B checkpoints with candidate helpfulness reward models (Skywork-Reward-V2, Athene, ArmoRM
> helpfulness head) and harmlessness signals (ArmoRM safety head, Qwen3Guard, and if needed a downloaded
> Safe-RLHF cost model or Llama-Guard); pick the helpful-harmless pair with the most negative Spearman that
> still resolves (base separable from a known-worse policy, CI excluding zero) and has base headroom, and lock
> it. Phase 2 (only if a pair qualifies): build the multi-objective RONPO dataset on those prompts, train
> RONPO-OS (lead) plus full-expectation and stabilized top-mass and the averaging baselines SPPO-avg,
> INPO-avg, DPO, all symmetric, seed 42, mu=base anchor, collapse-hardened (anchor 0.3 to 0.5, kappa
> annealed, lr <= 2.5e-8, warmup >= 0.2), select every checkpoint on the full validation set among
> stability-gate passers, and measure the worst objective plus the avg-vs-worst Pareto, testing whether the
> averaging baselines sacrifice a minority objective while RONPO holds both above a higher floor. Pre-register
> and hash the pair, prompts, metric, and thresholds first, confirm once on a fresh disjoint split with an
> independent safety-aware judge, and if RONPO leads, upload it to a public HuggingFace repo and add the
> result to the paper, else report the honest null. Never touch the spent sealed split, keep the stability
> gate unchanged, and use available GPUs without touching another user's job.
