# CODEX PROMPT — Qwen3-8B RONPO-OS: Stage-1 check, then Stage-2 continuation, HF upload, Table 4

## 0. Goal and decision structure

**Goal.** Make RONPO lead the worst-objective comparison at Qwen3-8B, with the **objective-stratified (OS)**
estimator as the lead arm. OS won the 1.5B stage-2 estimator ablation (worst-objective OS `0.500` >
top-mass `0.484` > base `0.414`; `/ext_hdd/sjkim/mnpo/eval/ronpo_topmass_vs_os_stage2_20260716/`), and its
unbiased, variance-reduced target is now analyzed in `ronpo_aaai/main_v3.tex`, Appendix
`app:estimators`. This run tests whether that advantage holds at 8B.

**Two-phase decision (run Phase A first):**

- **Phase A — Stage-1 OS check.** Train / select RONPO-OS at Stage-1 and evaluate it against every baseline
  on the worst-objective protocol. **If OS beats the baselines, stop the search: upload the OS checkpoint to
  a public HuggingFace repo and add it to Table 4 of `main_v3.tex`.**
- **Phase B — Stage-2 continuation (only if Phase A does not win).** Continue from the Stage-1 checkpoints
  into Stage-2 for all methods symmetrically (OS, full-expectation, stabilized top-mass, and the baselines),
  re-evaluate, and apply the same upload + Table-4 rule to the best RONPO arm.

**This is confirmatory, not manufacturing.** If RONPO does not lead the pre-registered worst-objective
metric on a fresh split, report the honest result and change nothing. No metric/seed/split switching after
seeing a ranking; regenerate every reported number from JSON.

## 1. Constraints

1. **Continue from Stage-1.** Phase B resumes from existing Stage-1 8B checkpoints under
   `/NHNHOME/26msit001_A/BASE/AIPR/sjkim/revision_qwen3_8b/full_iter1/`; do not retrain Stage-1 from scratch.
2. **Deadline: tonight 21:00 KST (2026-07-16).** Sequence the work so a clean, honest deliverable exists by
   then even if Phase B is not finished: Phase A first (cheapest, may already settle it), then Phase B if
   time allows. If Phase B cannot finish symmetrically by 21:00, ship the Phase-A result and mark Phase B
   in progress; never ship an unfair partial comparison.
3. **HuggingFace public upload, minimal local storage.** Upload the winning model(s) to public repos, then
   delete large local artifacts you no longer need: optimizer states (`optimizer.pt`), intermediate
   checkpoints, and redundant merged shards. Keep only the final best checkpoint per reported method plus
   its eval JSON. Stream decode/score outputs to compact JSONL; do not accumulate multiple full-precision
   copies of any 8B model on local disk.
4. **RONPO arms include OS.** OS (`mnpo_scripts/build_os_ronpo_targets.py`, `target_os_k{K}`) is the lead
   arm. Phase B also runs full-expectation and top-mass; top-mass must be stabilized (anneal kappa
   soft-to-hard, stronger anchor and SFT weight, lower peak lr, longer warmup) so it passes the unchanged
   reward-blind stability gate, or it is dropped and that is logged.
5. **Use all available GPUs.** Authorized pools: 4 B200 GPUs, and the H200 `air3` project (4 GPUs) if a
   read-only check shows them free. Also use free GPUs on `odin2` (A100 80GB), but never GPU 0 there while
   another user's process holds it. Sample every pool read-only before launch; never stop or modify another
   user's job. Partition work across pools (for example decode and RM scoring on odin2/H200, training on
   B200).
6. **Goal is a genuine RONPO lead**, obtained on equal footing with the baselines. A RONPO win from tuning
   RONPO harder than the baselines does not count.

## 2. Method mechanics (do not re-derive, reuse the repo)

- Trainer `mnpo_scripts/mnpo_trainer.py` (`loss_type=ronpo`), config `mnpo_scripts/mnpo_config.py`. Keep
  `mu = base` as the fixed KL anchor (`--ref_model`); the Stage-2 opponent `pi_t` is the Stage-1 policy,
  supplied as `--history_paths` to `mnpo_scripts/precompute.py`.
- Pool build: `on_policy_data_gen/decode.py` (`enable_thinking=false`, seed 42), score with
  `on_policy_data_gen/rm_{skywork,athene,armo}.py`, then targets from
  `mnpo_scripts/build_multi_objective_dataset.py` and `mnpo_scripts/build_os_ronpo_targets.py`
  (`target_os_k{K}`, `target_fullexp_k{K}`, `target_topmass_k{K}`).
- Baselines share the same pool. The offline methods (DPO, IPO, SimPO) and the averaged self-play methods
  (SPPO-avg, INPO-avg) retrain once on the best/worst pair drawn from the current-stage pool; the iterative
  methods refresh their opponent to the Stage-1 policy. Every method advances the same number of stages.
- Stability gate: `scripts/revision/flagship/stability_gate_corrected.py` (unchanged, reward-blind).

## 3. Phase A — Stage-1 OS check

1. Select the best Stage-1 RONPO-OS checkpoint at 8B on the 128-prompt validation split by the §5 metric
   (reuse the OS runs from `ronpo_variant_search_20260715`, for example `r2_os_anneal_anchor030`, or train
   a clean OS Stage-1 arm if none passes the gate). Must pass the stability gate.
2. Select each baseline's best Stage-1 checkpoint the same way, matched budget.
3. Evaluate OS and all baselines with the worst-objective protocol (§5).
4. If OS has the highest worst-objective among all methods, go to §6 (upload + Table 4). Otherwise proceed
   to Phase B.

## 4. Phase B — Stage-2 continuation (symmetric)

1. Seed each method from its best Stage-1 checkpoint (§3).
2. Refresh one shared Stage-2 pool from the Stage-1 policy (decode, 3-RM score, build all target columns).
3. Precompute logps: `--ref_model` = base, `--history_paths` = the Stage-1 checkpoint that generated the
   pool.
4. Train Stage-2 for OS, full-expectation, stabilized top-mass, and every baseline, same seed and matched
   budget.
5. Select the best Stage-2 checkpoint per method on validation by the §5 metric. Log the full grid and W&B
   run IDs to `sweep/`.

## 5. Metric, confirmation, decision

- **Primary metric (pre-register and hash before ranking):** worst objective, defined as the minimum over
  objectives of the per-objective score against base, on the local-RM protocol (647 held-out prompts, 3
  RMs), with the reward-model-independent judge panel as the confirmatory signal. Report mean objective,
  per-objective deltas, cross-objective spread, and win rate against base alongside.
- **Evaluation harness:** the `scripts/run_stage2_controlled_eval_parallel.sh` pipeline (decode, merge,
  3-RM score, `mnpo_scripts/evaluate_multi_objective_models.py`), pointed at whichever GPU pool is free.
- **Fresh confirmation once** on a prompt-disjoint split; measured a single time, no peek-and-continue,
  never the spent sealed split (`results/p1_sealed_reward_seed42_20260714/`).
- **Decision:** best RONPO arm highest worst-objective among all methods -> upload + Table 4 (scope honestly,
  state if the margin vs base is within noise); leads baselines but not base -> report as best trained
  method, upload, scope; otherwise honest null, no upload, no Table-4 edit.

## 6. HuggingFace upload and Table 4

- Push the winning model(s) to **public** repos under the project org; model card records base model,
  method and key hyperparameters, the pre-registered metric, the fresh-split result with CIs, and the honest
  scope. Record repo URL and commit. Then clean local large files per constraint 3.
- Regenerate the Table-4 fragment from the fresh `model_summary.json` via a committed script (mirror
  `analysis/ronpo_8b_reconstruction_20260714/build_*table*.py`). RONPO on top with CIs shown; keep every
  baseline row and objective. Rebuild `ronpo_aaai/main_v3.tex` with TinyTeX (0 fatal, 0 unresolved refs and
  cites, 0 overfull); record SHA-256 of the final `.tex` and `.pdf`.

## 7. Deliverables (into `results/p1_8b_stage2_20260716/`)

- `PREREG.md` + `metric_lock.json` (+SHA); Stage-2 pool manifest (+SHA) and RM scores if Phase B runs.
- `sweep/` validation grid (all arms, stage-tagged) + W&B IDs.
- `fresh/model_summary.json` + per-objective CSVs; the committed aggregation and table script.
- `REPORT.md` with the Phase-A / Phase-B outcome and the primary plus all secondary numbers.
- On a win: HF repo URLs + commits; regenerated Table-4 fragment; rebuilt `main_v3.pdf`; SHAs; a note of
  which large local files were deleted.
- `COMPLETION_AUDIT.md` + `fix_log.md`, ending `spent_sealed_split_touched=false`.

## 8. One-paragraph brief (paste at top of the codex run)

> Make RONPO lead the worst-objective comparison at Qwen3-8B with the objective-stratified (OS) estimator as
> the lead arm, by 21:00 KST tonight. First run Phase A: select or train Stage-1 RONPO-OS at 8B, evaluate it
> against every baseline on the local-RM worst-objective protocol (647 prompts, 3 RMs) plus the independent
> judge, and if OS beats the baselines, upload it to a public HuggingFace repo and add it to Table 4 of
> ronpo_aaai/main_v3.tex. If OS does not win at Stage-1, run Phase B: continue from the Stage-1 checkpoints
> into Stage-2 for all methods symmetrically (keep mu=base as the KL anchor, set the opponent to the Stage-1
> policy via --history_paths, refresh one shared pool, and train OS, full-expectation, and stabilized
> top-mass alongside SPPO-avg, INPO-avg, HT-MNPO, DPO, IPO, SimPO), then apply the same upload and Table-4
> rule to the best RONPO arm. Pre-register and hash the worst-objective metric before ranking, confirm once
> on a fresh prompt-disjoint split, and report honestly if RONPO does not lead. Use all available GPUs (4
> B200, the H200 air3 project if free after a read-only check, and free odin2 GPUs, never another user's
> GPU), upload models to public HF repos and delete large local files (optimizer states, intermediate
> checkpoints) after upload, and never touch the spent sealed split.
