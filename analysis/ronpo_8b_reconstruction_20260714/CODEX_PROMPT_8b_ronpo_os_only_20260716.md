# CODEX PROMPT — Qwen3-8B: add a stabilized RONPO-OS row to Table 4 by retraining ONLY RONPO-OS

## 0. Scope and rationale

Table 4 of `ronpo_aaai/main_v3.tex` is the Qwen3-8B worst-objective diagnostic
(`tab:qwen3-robust-validation`). Its baseline policies are already trained and already pass the stability
gate. **Do not retrain the baselines.** The only missing, and the only unstable, arm is **RONPO-OS**. So
this run retrains **only RONPO-OS**, stabilizes it, evaluates it against the existing baseline set on the
same protocol, and if it leads, adds one row to Table 4 and uploads the model.

Why only OS: the baselines are at their published best under the Stage-1 budget, and the prior failure was
specific to the OS checkpoint collapsing (a weak-anchor, knife-edge repetition instability), not to the
baselines. Retraining the whole fleet was unnecessary and previously blocked on SimPO. Reusing the frozen
baselines is both cheaper and fair.

**Fairness rule:** OS may use only the *stability* hardening already applied to the other RONPO arms
(reference anchor, kappa annealing, lr, warmup, preference-SFT). Do not give OS extra *performance* tuning
that the baselines did not get. Baselines stay exactly as trained.

## 1. Hard rules

- Never decode, score, tune, or select on the spent 604-prompt sealed split
  (`results/p1_sealed_reward_seed42_20260714/`).
- Pre-register and hash the worst-objective metric and the stability gate before any ranking.
- Keep the stability gate `scripts/revision/flagship/stability_gate_corrected.py` unchanged and reward-blind.
- Regenerate every reported number from JSON. Report all per-objective outcomes. If OS does not lead, report
  the honest result and change nothing.

## 2. Reuse the existing baseline set (no retraining)

- Baselines and base: reuse the existing gate-passing Stage-1 8B checkpoints and, where available, their
  existing generations and RM scores (the frozen candidates used by
  `results/p1_8b_stage2_20260716/` and the variant-search calibration: `base`, `sppo_avg`, `inpo_avg`,
  `ipo`, `dpo`, `ht_mnpo_{helpfulness,safety,conciseness}`, and the RONPO `top-mass` / `full-expect` arms).
- The Table-4 baseline set is whatever the paper's `tab:qwen3-robust-validation` already reports; match that
  exact set. If a baseline row in Table 4 has an existing gate-passing checkpoint, reuse it verbatim. Do not
  re-decode a baseline whose generations already exist unless the prompt set changed.
- If a baseline in the Table-4 set has no gate-passing checkpoint at all, keep the paper's current handling
  of it (exclude or its existing best step); do not launch a fleet retrain to fix it.

## 3. Retrain ONLY RONPO-OS, hardened against collapse

Train RONPO-OS from the Stage-1 base pool (same pool, seed 42, same step budget as the existing RONPO arms).
Harden by construction so the selected checkpoint is robust across steps, not knife-edge:

- `reference_anchor_weight` in {0.3, 0.4, 0.5} (never 0.1); `preference_sft_weight` in {0.02, 0.03, 0.05}.
- Kappa annealed soft to hard (e.g. `target_os_k0p05 -> k0p02 -> k0p01 -> k0p007 -> k0p005`) via
  `mnpo_scripts/build_os_ronpo_targets.py`, rather than a fixed sharp kappa.
- Peak learning rate at or below `2.5e-8`, warmup ratio at or above `0.2`.
- Keep mu = base as the KL anchor.

**Selection (the fix that was missing before):** save every 100 steps, and **gate every saved OS checkpoint
on the full 647-prompt held-out set** (decode with `scripts/revision/flagship/decode_vllm_non_thinking.py`,
seed 42, temperature 0.7, top-p 0.9, 4096 max new tokens, bf16, non-thinking template). Select the best
worst-objective OS checkpoint **only among 647-gate passers**, and prefer a step whose neighbors also pass
(robust, not a lucky step). Never select on the 128-prompt gate. If no OS checkpoint passes the 647 gate
after the hardening sweep, report that honest negative and stop.

## 4. Evaluate and decide

- Decode OS on the same 647 held-out prompts (and the fresh confirmation split). Reuse base and baseline
  generations and their RM scores; score only the new OS responses with Skywork, Athene, ArmoRM.
- Re-aggregate the worst-objective metric over the combined set {base, existing baselines, OS} with
  `mnpo_scripts/evaluate_multi_objective_models.py` (normalization is per-prompt over the model pool, so it
  must be recomputed once OS is added). Report Avg, Worst, per-objective deltas, disparity, win rate vs base,
  and the independent judge panel as the confirmatory signal.
- **Fresh confirmation once** on a prompt-disjoint split, never the spent split, no peek-and-continue.
- **Decision:** OS has the highest gate-passing worst-objective among all models -> upload + add the OS row
  to Table 4, scoped honestly (state if the margin vs base is within noise). OS leads all baselines but not
  base -> report as best trained method and add the row with that scoping. Otherwise honest null, no upload,
  no Table-4 edit.

## 5. Upload, storage, GPUs

- On a win, push only the OS checkpoint to a **public** HuggingFace repo with an honest model card (base,
  RONPO-OS + hyperparameters, metric, fresh-split result with CIs, scope). Record repo URL and commit.
- Minimize local storage: after upload, delete OS optimizer states and intermediate checkpoints; keep only
  the final OS checkpoint plus eval JSON. Do not duplicate baseline weights.
- Use available GPUs (4 B200, H200 `air3` if free after a read-only check, free odin2 GPUs), never another
  user's GPU. This run is small: one training sweep for OS plus one decode-and-score pass for OS only.

## 6. Table 4 edit (only on a win)

Regenerate the `tab:qwen3-robust-validation` fragment from the fresh `model_summary.json` via the committed
script (mirror `analysis/ronpo_8b_reconstruction_20260714/build_*table*.py`). Add the OS row with CIs shown;
keep every existing baseline row and the honest underpowered-diagnostic caveat already in the caption unless
the new evaluator is demonstrably powered. Rebuild `ronpo_aaai/main_v3.tex` with TinyTeX (0 fatal, 0
unresolved refs and cites, 0 overfull); record SHA-256 of the final `.tex` and `.pdf`.

## 7. Deliverables (into `results/p1_8b_ronpo_os_only_20260716/`)

- `PREREG.md` + `metric_lock.json` (+SHA); stability-gate spec + hash.
- `sweep/` OS per-step 647-gate results and validation metrics + W&B IDs; note which baseline checkpoints and
  scores were reused (paths + hashes).
- `fresh/model_summary.json` + per-objective CSVs; the committed aggregation and table script.
- `REPORT.md` with the decision, primary and all secondary numbers, and the OS checkpoint's per-step
  stability profile.
- On a win: HF repo URL + commit; regenerated Table-4 fragment; rebuilt `main_v3.pdf`; SHAs; deleted-files
  list.
- `COMPLETION_AUDIT.md` + `fix_log.md`, ending `spent_sealed_split_touched=false`.

## 8. One-paragraph brief (paste at top of the codex run)

> Add a stabilized RONPO-OS row to Table 4 (`tab:qwen3-robust-validation`, the Qwen3-8B worst-objective
> diagnostic) of ronpo_aaai/main_v3.tex by retraining ONLY RONPO-OS. Do not retrain the baselines: reuse the
> existing gate-passing Stage-1 8B baseline checkpoints, generations, and RM scores that already build Table
> 4. The previous failure was specific to the OS checkpoint collapsing (weak anchor 0.1, sharp kappa,
> knife-edge repetition, and selection on only 128 prompts), so fix exactly that: retrain OS with a stronger
> reference anchor (0.3 to 0.5), soft-to-hard kappa annealing, peak lr at or below 2.5e-8, warmup at or above
> 0.2, and a nonzero preference-SFT term, keep mu=base as the KL anchor, gate every saved OS checkpoint on
> the full 647-prompt held-out set, and select only among 647-gate passers (never 128). Decode OS on the 647
> prompts and the fresh split, reuse base and baseline generations and scores, score only the new OS
> responses, and re-aggregate the worst-objective metric over {base, existing baselines, OS}. OS may use only
> the same stability hardening the other RONPO arms already use, not extra performance tuning. Pre-register
> and hash the metric, confirm once on a fresh prompt-disjoint split with the independent judge, and if OS
> leads, upload only the OS checkpoint to a public HuggingFace repo and add its row to Table 4, else report
> the honest null. Never touch the spent sealed split, keep the stability gate unchanged, and delete OS
> optimizer states and intermediate checkpoints after upload.
