# CODEX PROMPT — Qwen3-8B Stage-1 retrain, collapse-hardened, RONPO-OS lead

## 0. Why the previous run failed (fix these exactly)

The prior Stage-1/Stage-2 attempt returned an honest FAIL because the selected RONPO-OS checkpoint
collapsed. Root cause, verified from artifacts:

- The OS config `r1_os_stratified_k002` (reference_anchor_weight `0.1`, kappa `0.02`) is **latently unstable**:
  across training steps it oscillates between clean and catastrophic repetition. On the 128-prompt gate,
  steps 200/500/600/800 fail with max repeat runs 1501/1518/601/1702, while 100/300/400/700/900 pass with
  runs of 4 to 6. The instability is **prompt-specific**.
- Selection used the **128-prompt** gate. Step 900 passed there (run 4), but on the **647-prompt** gate the
  same checkpoint failed with a repeat run of 587 on one prompt. A 128-prompt gate is too small to catch the
  collapse triggers, so it let a latently collapsed checkpoint through.
- Phase B could not run because **SimPO had no stable Stage-1 parent** (all SimPO candidates failed the gate),
  so a symmetric comparison was impossible.

**Three fixes are mandatory in this run:**
1. **Never select on 128 prompts.** Gate every saved checkpoint on the full held-out set (647 prompts, or
   larger if available) and select only among checkpoints that pass that gate.
2. **Harden RONPO-OS (and top-mass) against repetition by construction**, not by lucky steps: stronger
   reference anchor, soft-to-hard kappa annealing, lower peak learning rate, longer warmup, and a nonzero
   preference-SFT term. The passing config must be robust across steps, not knife-edge.
3. **Produce a stable Stage-1 parent for every method, including SimPO**, so a symmetric set exists.

## 1. Goal

Retrain all methods from Stage-1 at Qwen3-8B with collapse-hardened settings, with **RONPO-OS as the lead
arm**, then evaluate worst-objective. If the best RONPO arm (OS preferred) beats the baselines on a
647-prompt-gate-passing, held-out worst-objective comparison, upload it to a public HuggingFace repo and add
it to Table 4 of `ronpo_aaai/main_v3.tex`. Otherwise report the honest result and change nothing. This is
confirmatory: no metric/seed/split switching after seeing a ranking, no fabricated numbers.

## 2. Hard rules

- Never decode, score, tune, or select on the spent 604-prompt sealed split
  (`results/p1_sealed_reward_seed42_20260714/`).
- Pre-register and hash the worst-objective metric and the stability gate before any ranking.
- Symmetric budget: every method gets the same step budget and the same hardening opportunity. A RONPO win
  from tuning RONPO harder than the baselines does not count.
- Keep the stability gate `scripts/revision/flagship/stability_gate_corrected.py` unchanged and reward-blind.
  Do not weaken it to pass a degenerate model. Hardening happens in training, not in the gate.
- Report all per-objective outcomes and all arms, including failures.

## 3. Collapse-hardened Stage-1 training (from base, seed 42)

Train each method from the base policy on the Stage-1 pool. Keep mu = base as the KL anchor.

**RONPO arms (three targets, all hardened):**
- **RONPO-OS** (`target_os_k{K}`, `mnpo_scripts/build_os_ronpo_targets.py`) is the lead arm.
- Full-expectation and top-mass are also trained.
- Hardening grid for the RONPO arms (sweep, pick the most stable that still moves worst-objective):
  - `reference_anchor_weight` in {0.3, 0.4, 0.5} (not 0.1), `preference_sft_weight` in {0.02, 0.03, 0.05}.
  - Kappa annealed soft to hard, for example `target_os_k0p05 -> k0p02 -> k0p01 -> k0p007 -> k0p005` over
    training, rather than a fixed sharp kappa.
  - Peak learning rate at or below `2.5e-8`, warmup ratio at or above `0.2`.
  - `ronpo_alpha` in {0.1, 0.15}, `ronpo_tau` {0.1}.

**Baselines (symmetric, each must yield a gate-passing Stage-1 parent):**
- SPPO-avg, INPO-avg, HT-MNPO (help / safety / conciseness), DPO, IPO, SimPO.
- **SimPO is the most fragile (reference-free); harden it explicitly**: raise `simpo_gamma`, lower learning
  rate, longer warmup, and if needed a light reference-anchor or SFT term, until at least one SimPO Stage-1
  checkpoint passes the 647-prompt gate. If, after a symmetric hardening budget, SimPO still cannot pass,
  record that and proceed with the largest symmetric set that does pass, stating the exclusion explicitly.

## 4. Checkpoint selection (the key fix)

1. Save checkpoints every 100 steps.
2. **Gate every saved checkpoint on the full 647-prompt held-out set** (decode with the main-guarded
   `scripts/revision/flagship/decode_vllm_non_thinking.py`, seed 42, temperature 0.7, top-p 0.9, 4096 max new
   tokens, bf16, non-thinking template). Discard any checkpoint that fails the repetition gate.
3. Among the gate-passing checkpoints of each method, select the one with the best worst-objective on the
   pre-registered validation metric. Never select a checkpoint that only passed a smaller gate.
4. Log the full per-step gate results and validation metrics to `sweep/`, so the stability profile of the
   selected step is auditable (it must pass on 647, and ideally its neighbors should pass too).

## 5. Worst-objective evaluation and decision

- **Metric (pre-registered):** worst objective as the minimum over objectives of the per-objective score vs
  base, on the local-RM protocol (647 prompts, Skywork + Athene + ArmoRM;
  `scripts/run_stage2_controlled_eval_parallel.sh` pipeline and
  `mnpo_scripts/evaluate_multi_objective_models.py`), with the reward-model-independent judge panel as the
  confirmatory signal. Report mean objective, per-objective deltas, disparity, and win rate vs base.
- **Fresh confirmation once** on a prompt-disjoint split, measured a single time, never the spent split.
- **Decision:** best RONPO arm (OS preferred) has the highest gate-passing worst-objective among all methods
  -> upload + Table 4, scoped honestly (state if the margin vs base is within noise). Leads baselines but not
  base -> report as best trained method, upload, scope. Otherwise honest null, no upload, no Table-4 edit.

## 6. Upload, storage, GPUs

- On a win, push the winning model(s) to **public** HuggingFace repos with an honest model card (base, method,
  hyperparameters, metric, fresh-split result with CIs, scope). Record repo URL and commit.
- Minimize local storage: after upload, delete optimizer states and intermediate checkpoints; keep only the
  final best checkpoint per reported method plus eval JSON. Do not keep multiple full-precision copies of any
  8B model on local disk.
- Use all available GPUs: 4 B200, the H200 `air3` project (4 GPUs) if a read-only check shows them free, and
  free `odin2` GPUs (never GPU 0 there while another user holds it). Sample every pool read-only before
  launch; never stop or modify another user's job.

## 7. Deliverables (into `results/p1_8b_stage1_retrain_stable_20260716/`)

- `PREREG.md` + `metric_lock.json` (+SHA); the stability-gate spec and its hash.
- `sweep/` per-method, per-step gate results on 647 prompts and validation metrics + W&B IDs.
- `fresh/model_summary.json` + per-objective CSVs; the committed aggregation and table script.
- `REPORT.md` with the decision, the primary and all secondary numbers, and the per-step stability profile of
  each selected checkpoint.
- On a win: HF repo URLs + commits; regenerated Table-4 fragment; rebuilt `main_v3.pdf` (0 fatal, 0 unresolved
  refs and cites, 0 overfull); SHAs; list of deleted large local files.
- `COMPLETION_AUDIT.md` + `fix_log.md`, ending `spent_sealed_split_touched=false`.

## 8. One-paragraph brief (paste at top of the codex run)

> Retrain all methods from Stage-1 at Qwen3-8B with collapse-hardened settings and RONPO-OS as the lead arm.
> The previous run failed because the OS config with a weak reference anchor (0.1) and sharp kappa (0.02) is
> latently, prompt-specifically unstable: it oscillates between clean and catastrophic repetition across
> steps, and selection on only 128 prompts let a checkpoint through that then failed the 647-prompt gate with
> a repeat run of 587, while SimPO had no stable Stage-1 parent at all. Fix this by (1) gating every saved
> checkpoint on the full 647-prompt held-out set and selecting only among 647-gate passers, never on 128; (2)
> hardening the RONPO arms by construction with a stronger reference anchor (0.3 to 0.5), soft-to-hard kappa
> annealing, peak lr at or below 2.5e-8, warmup at or above 0.2, and a nonzero preference-SFT term, so the
> passing checkpoint is robust across steps rather than knife-edge; and (3) explicitly hardening SimPO
> (reference-free, most fragile) until it yields a gate-passing Stage-1 parent so a symmetric comparison
> exists. Keep mu=base as the KL anchor, keep the stability gate unchanged and reward-blind, pre-register and
> hash the worst-objective metric, select on validation, confirm once on a fresh prompt-disjoint split with
> the independent judge, and if the best RONPO arm (OS preferred) leads, upload it to a public HuggingFace
> repo and add it to Table 4 of ronpo_aaai/main_v3.tex, else report the honest null. Use all available GPUs
> without touching another user's job, delete large local files after upload, and never touch the spent
> sealed split.
