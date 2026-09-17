# Campaign code snapshot, 2026-09-17 (current)

Byte-identical copy of `/work/sub_20260914/code` on the campaign pod, plus the
dependency-queue controller from `/work/uf4_20260910/code/queue_controller.py`.
Every file here hashes the same as its source at the time of the copy; the
sha256 prefix is recorded per file in the table at the end.

This directory is the **current** state of the pipeline. The sibling
`code_snapshot_20260916/` is left frozen on purpose: it records the exact code
that produced the per-prompt (prompt-wise) NBPO numbers reported at that date,
and several of its files have since been edited here, so the two disagree by
design. Where a file appears in both, the 20260916 hash is the provenance of
the published number and the hash here is the code as it now stands.

## Corrected tensor roles (2026-09-18)

An external implementation audit found that this scorer wired the game tensors
against the paper's own definition, and the fix is the most consequential change
in this directory:

    was:  A_policy <- learner-learner      A_ref <- learner-reference
    now:  A_policy <- learner-reference    A_ref <- reference-reference

Section 5.2 defines the policy game on the learner bank against the comparator
bank, and `compute_disagreement_point` documents its own argument as "centered
payoffs of reference responses (as learner) against reference comparators". The
old wiring satisfied neither. It is not a naming slip: with every learner tied
to every other learner, every reference tied to every other reference, and every
learner beating every reference at .75, the paper's wiring gives policy value
+.25, disagreement 0 and surplus +.25, while the old wiring gives 0, +.25 and
surplus **-.25**. The sign of the surplus flips, and the surplus is what
finite-pool feasibility tests, so the solver was solving a different finite
problem exactly rather than the declared one approximately.

`A_LL` also cannot carry the policy game: it is antisymmetric with a zero
diagonal, so a symmetric strategy scores identically zero against itself and the
surplus carries almost no signal about the learner bank — consistent with the
near-zero target correlations the UW and US rounds measured.

`union_score_panel.py` now writes `A_LL`, `A_LR` and `A_RR` each under its own
name and aliases `A_policy`/`A_ref` to `A_LR`/`A_RR`, so an existing reader gets
the paper's game unchanged while a pair-label consumer can ask for the learner
block explicitly. `build_panel_softlabels.py` was reading `A_policy` and
expecting the learner block, so it now reads `A_LL` and refuses a score
directory written before this fix instead of silently taking the cross block.
The reference triangle was previously kept only as a scalar diagnostic and is
now required for a prompt to be retained, because `d` is not defined without it.

`prosper_targets_panel.py` separates the roles itself from the raw judgments and
is unaffected. Re-scoring needs no new judging: all three role blocks were
already fully judged on every panel.

Serialization changed with it. `quantize_canonical_row` used to round both solver
masses to ten FIXED decimals and rebuild the target from the rounded values; a
per-prompt Nash solve concentrates mass down to the 1e-12 probability floor, and
1e-12 at ten decimals is exactly 0, which the canonical validator rejects. Fixed
decimals are the wrong instrument for a quantity spanning twelve orders of
magnitude, so masses are now written at exact float64 round-trip precision and
the target is rebuilt from those same doubles. The identity holds to ~1e-16,
inside the 1e-9 gate, with the certified solution intact. Rows written before
this carry `canonical_target_quantized_decimals = 10`; new rows carry
`canonical_target_serialization = "exact_float64_round_trip"`.

## Follow-up audit, 2026-09-18 (b949919)

A second audit pass on the first round of fixes found three of them incomplete
and three new defects in the style-control code added with them. All six are
closed here.

**The scorer fix reached only the generalized scorer.** `union_score_uw.py`,
which wrote the published UW scores, still carried the old wiring, and the
solver's loader accepted a score directory on hash and shape alone — a sha256
certifies that the bytes are the ones that were written, not that they mean what
the reader assumes. Score shards now carry `tensor_role_schema` and per-tensor
`bank_ids`; the solver refuses a shard that does not declare the schema this
game needs, and `union_score_uw.py` refuses to run at all unless explicitly
acknowledged. A directory written before 2026-09-18 has no schema and is
rejected rather than solved into a different finite game.

**The serializer fix reached only uw1 and uw1c.** The us1, ut1 and uw3 modules
had been generated from the pre-fix base, so they still rounded masses to ten
fixed decimals: a 1e-12 mass went to 0 and the target drifted by .399. Every
active solver, the two legacy shared-weight ones included, now serializes at
exact float64 precision, and the acceptance fixture is parameterized across all
seven.

**The pool digest hashed names, not content.** `train_pool_sha256` and
`dev_pool_sha256` were both `object_hash(sorted(outputs))` — the split NAMES —
so they were equal to each other and unmoved by the pool. They are now a digest
over each split's own rows (prompt id, role, occurrence index, candidate id,
response sha256); mutating one byte of one response changes it, and a change
outside the split does not. The hardcoded `panel: UF-4` is rewritten per panel
by the generator.

**The style-control length feature compared tokens with words.** The arm's
length came from its stored token count and the baseline's from
`len(text.split())`, so two identical answers differed by .44 on the feature the
control exists to remove. One frozen tokenizer now measures both sides, an
identical-text self-difference of zero is asserted before any data is read, and
the corrected numbers replace the ones computed on 2026-09-17. The fit reports
whether it converged and the bootstrap discards a replicate that did not; a
prompt enters only when both presentation orders parsed.

**The generator's own guard was a false positive.** It rejected any generated
module whose `--help` contained the substring `uw1`, which fires on `uw1c` —
correctly repointed — after the files had already been written. It now compares
whole tokens and exempts the panel being generated.

## Third audit pass, 2026-09-18 (97e03c0)

Six of the previous round's items were confirmed resolved. Four remained, and
all four are closed here.

**A04, the one that mattered: Algorithm 1's acceptance step was never wired into
the prompt-wise training path.** The traced path was job generator —
`torch.distributed.run` — `run_mnpo` — `train()` — `save_model()`, with
nothing between or after it that evaluated, rejected or promoted a candidate. A
certified solver residual does not substitute: it certifies the finite target,
not the policy fitted to it. `nbpo_local_gate.py` now performs that step. It
builds ONE REPRESENTATION PER PROMPT — the batched `game_values` divides by
the prompt count, so a batch containing +.2 and -.1 averages to +.05 and passes
while the compromise failed on one of those prompts — and applies three
declared checks: every surplus finite, the fraction of development prompts
positive on every objective at least `--coverage-min`, and the held-out nMSE at
most `--nmse-max`. On a pass it writes a versioned accepted directory carrying
the checkpoint's fingerprint; on a failure it promotes nothing and returns the
parent. `scripts/nbpo/eval_game_value.py` stays global and now says so in its
own docstring, because global is correct for the Global Nash control.

The gate exists from 2026-09-18 and is not retroactive. Every number already
reported came from an ungated projection, and a later gate cannot change that;
anything described as gated must name a promotion record this file wrote.

**C01: the schema tag was a declaration with nothing behind it.** A shard could
say `lr_rr_v2` while its `A_policy` was not its own `A_LR`. The loader now checks
the aliases against the semantic arrays, refuses a shard whose `A_policy` equals
its `A_LL` (the pre-fix wiring under a post-fix label), and checks the recorded
bank ids.

**C02: the pool digest trusted a cached hash.** A row whose response text was
edited while its `response_sha256` was left behind produced an unchanged digest.
`split_pool_digest` now recomputes from the text, and the refusal lives in a
separate `verify_pool_row_digests` that the solve calls first — so the digest
stays a pure function of content while an internally inconsistent pool still
stops the run.

**C03: the UW1 base still labelled itself UF-4.** It is `UW1` now, and the
generator's substitution anchor follows.

## The prompt-wise NBPO line

The prompt-wise change is in the solver, not the trainer: NBPO's dual
multipliers are fitted separately at each prompt instead of being shared across
prompts. `solve_pros4_targets.py` is the shared-weight solver, and every
per-prompt and per-panel variant is *generated* from it by a single generator,
so the arms cannot drift apart in anything except the rule they implement.

| Arm | Representation | Aggregation | Weight scope |
|---|---|---|---|
| NBPO-PW | adaptive game | Nash | per prompt |
| Fixed-reference Nash-PW | fixed reference | Nash | per prompt |
| PROSPER | adaptive game | absolute max-min | per prompt |
| NBPO (shared weights) | adaptive game | Nash | one vector for all prompts |
| MOPO | adaptive game | importance-weighted cloning | per prompt |

- `build_pw_solvers.py` — generates the three per-prompt solvers from the
  pristine global solver, with `count == 1` assertions on every anchor.
- `build_panel_solvers.py` — the same generator for the US/UT/UW panels. It
  also rewrites the hard-coded `K = 4` to `K = len(base.OBJECTIVES)` and then
  imports every module it generated to assert the objective count agrees. A
  generated probe that kept `K = 4` on a two-objective panel refused every
  prompt with "beta must be a positive vector of length K", which reads as
  infeasibility rather than as a bug; that is why the assertion exists.
- `solve_pros4_targets.py`, `solve_pros4_targets_v2.py` — the shared-weight
  solvers the rest are generated from.
- `solve_pros4_pw_nbpo*.py`, `solve_pros4_pw_fixedref*.py`,
  `solve_pros4_prosper*.py` — per-prompt Nash, fixed-reference Nash and
  absolute max-min, per round and per panel.
- `solve_mopo_targets_us1.py`, `solve_mopo_targets_ut1.py` — MOPO importance
  weights, which are per-candidate rather than per-pair.
- `*_uw1c.py` — the corrected UW round: the same rules on the re-scored
  tensors, reading `scores/uw1c` and `splits/uw_v1cp` so the published `uw1`
  artifacts are never overwritten.
- `*_uw3.py` — the re-decoded UW pool round (T .8, top-p .9, 2,048 tokens),
  reading `scores/uw3` and `splits/uw_v3p`.

## Panel construction

- `union_freeze_uw.py`, `union_freeze_us.py`, `union_freeze_ut.py` — freeze one
  panel's prompt split, candidate pool and objective list.
- `union_judge_uw.py` — pool judging for the UW panel.
- `union_score_uw.py`, `union_score_panel.py` — Bradley-Terry fits and the
  frozen teacher record. `union_score_panel.py` is the generalized form and
  reproduces the UW tensors exactly: max absolute difference 0.0 over 337,920
  floats against the published UW artifacts.
- `build_panel_softlabels.py` — pair probabilities for the soft-label losses.
- `prosper_targets.py`, `prosper_targets_panel.py` — PROSPER-format targets;
  the panel form reproduces the published pair-file hashes byte for byte.
- `build_prosper_dataset.py`, `panel_mopo.py`, `panel_stage2.py` — dataset
  materialization and the stage-2 driver. `panel_mopo.py` writes both the
  expected dataset manifest and solver-artifact hashes, so the trainer's
  provenance pins hold for the MOPO arm instead of being switched off.

## Feasibility and representability

`probe_feasible.py`, `probe_feasible_us1.py`, `probe_feasible_ut1.py`,
`probe_feasible_uw1.py` certify each prompt for each rule and write the
surviving split (the report is written before any raise, so a failure still
leaves a per-split record). `restrict_common_certified.py` and `restrict_v2.py`
intersect the arms onto one prompt set; `filter_prompt_level.py` and
`filter_and_materialize.py` apply the representability filter;
`clear_partial_targets.py` removes target directories without `complete.json`;
`probe_surrogate_cost.py` prices the surrogate.

## Queueing and training

- `queue_controller.py` — the dependency queue: specs under
  `jobs/queue/<priority>_<job_id>.json`, ascending `(priority, job_id)`, DONE
  never re-runs, FAILED/BLOCKED re-run only when `spec_sha256` changes, free
  devices held for the top blocked 4-GPU job. `load_specs` skips a spec that
  vanishes mid-poll, because renaming one used to kill the controller with
  `FileNotFoundError`.
- `exp2_pipeline.py`, `exp2_stage2.py` — the scaled round's two stages.
- `make_pros_train_jobs.py`, `record_and_train.py`, `make_train_v2.py`,
  `make_splits_and_queue.py`, `regen_configs.py`, `queue*.py`,
  `terminal_prosper.py` — per-arm configs and queue specs.

## Evaluation

- `build_eval_panels.py` — freezes the Arena-Hard v0.1 and AlpacaEval 2.0
  panels and their static baselines.
- `build_eval_panel_ahv2.py` — the same for Arena-Hard v2.0: 750 prompts
  (`hard_prompt` 500 + `creative_writing` 250), 26 languages, gpt-4.1 baseline.
- `gen_candidates.py` — greedy generation for evaluation.
- `judge_eval_pairwise.py`, `judge_eval_batch.py`, `judge_eval_batch_ahv2.py` —
  pairwise judging against the static baselines, both orders, ties 0.5.
- `mtbench_generate.py`, `mtbench_judge.py`, `mtbench_judge_batch.py` —
  MT-Bench two-turn generation and single-answer grading.
- `panel_eval_judge.py`, `panel_eval_matrix.py`, `panel_eval_selftest.py`,
  `setup_panel_eval.py`, `collect_panel_cells.py` — held-out panel evaluation.
  `panel_eval_matrix.py` averages a prompt's two judge orders and drops a
  prompt that has only one of them; an earlier version indexed by prompt alone
  and silently kept whichever order was written last, which put the base model
  at .359 against another draw of itself instead of .5. The self-test passed
  that bug because its synthetic rows only ever carried order 0, so it now
  builds both orders.
- `xplay_judge.py`, `xplay_matrix.py` — arm-versus-arm cross-play.
- `ahv2_style_control.py` — the style-controlled Arena-Hard v2.0 win rate, read
  off judgments already in hand. A per-arm logistic fit of the judged outcome on
  the normalized answer-length, header, bold-span and list-item differences
  against the frozen baseline, evaluated at zero style difference, with a
  2,000-replicate whole-prompt bootstrap that refits the regression in every
  replicate. It is NOT the official Arena-Hard style control — that fits one
  Bradley-Terry model over the whole arena under the official judge — and the
  output says so in its own `not_the_official_metric` field.
- `ahv2_sc_paired.py` — the paired bootstrap of the style-controlled DIFFERENCE
  between two arms. Two arms' unpaired intervals can overlap heavily and still
  separate, because they are read on the same prompts, so this resamples prompts
  once per replicate and refits BOTH regressions on that resample. An interval
  containing zero is reported as containing zero, never as equivalence.
- `upload_e2_arms.py`, `upload_uw_arms.py` — checkpoint upload.

## Statistics and diagnostics

- `paired_diff.py`, `paired_eval_diff.py`, `ahv2_paired.py` — whole-prompt
  paired bootstrap on the prompts fully parsed in both runs.
- `ahv2_breakdown.py` — Arena-Hard v2.0 by language and category.
- `length_control.py`, `lc_winrate.py` — the paired difference conditioned on
  response length, and win rate read at equal length.
- `abl_analysis.py` — the 2-draw versus 10-draw judgment-count ablation.
- `analyze_screen.py`, `analyze_wild.py`, `wild_selftest.py` — cycle,
  Condorcet and disagreement statistics.
- `sweep_l1.py`, `sweep_diag.py` — the multiplier-norm sweep and the learning
  rate ladder. `sweep_diag.py` parses residuals only after the last launch
  banner, so a killed run's stale log is not reported as live progress.
- `diag_pool_kl.py`, `panel_solver_diag.py`, `uw_solver_diag.py`,
  `factorial_2x2.py` — pool drift, solver certificates and the 2x2 factorial.

## Earlier panels

`generate_pros_pool.py`, `judge_pool_prosper.py`, `judge_pool_items.py`,
`judge_pool_pairs.py`, `judge_pairs.py`, `score_pros_pool.py`, `merge_ext.py`,
`unify_scoring.py`, `freeze_uf_panel.py`, `freeze_wild_panel.py`,
`judge_wild_items.py`, `freeze_safe_panel.py`, `freeze_safe_splits.py`,
`judge_fresh_safe.py`, `score_safe_pool.py`, `build_dpo_safe.py`,
`make_safe_solvers.py`, `make_safe_solvers2.py`, `solve_safe_targets.py`,
`solve_safe_prosper.py`, `solve_pros.py`, `gate_uf_arms.py`, `cc_split.py`,
`commit_split.py` belong to the PROSPER-setting, wild-checklist and SafeRLHF
rounds that preceded the union panels.

## One-off maintenance

`fix_cal.py`, `fix_dataset_env.py`, `fix_dev_shards.py`, `fix_dpo_dataset.py`,
`fix_target_manifest.py`, `patch_prosper.py`, `patch_solver_precision.py`,
`repoint_cc.py`, `repoint_solvers.py`, `redispatch4.py`, `redispatch5.py`,
`requeue_solves.py`, `clean_targets.py` are single-use repair scripts kept for
the record. They are not pipeline steps and re-running them is not meaningful.

## Not included

The deployed trainer lives in `mnpo_scripts/` at the repository root, not here.
Pool candidates, judgments, solver artifacts, datasets and checkpoints stay on
the pod and in the Hugging Face repositories; only code is snapshotted.

## File hashes

| File | sha256 (16) |
|---|---|
| `abl_analysis.py` | `a1ae9271f1a860e1` |
| `ahv2_breakdown.py` | `d4e97430d4f537bb` |
| `ahv2_paired.py` | `85038b4b314c5943` |
| `ahv2_sc_paired.py` | `9906157bc602e742` |
| `ahv2_style_control.py` | `ee230bf19864ed5a` |
| `analyze_screen.py` | `841409da20c363de` |
| `analyze_wild.py` | `f1226378f4072081` |
| `build_dpo_safe.py` | `f39441a6d0be257f` |
| `build_eval_panel_ahv2.py` | `63bbabb8ba9a4c97` |
| `build_eval_panels.py` | `8183ed5efbec9580` |
| `build_panel_softlabels.py` | `567ef4618851beb0` |
| `build_panel_solvers.py` | `d958a4ffd5bc3a4b` |
| `build_prosper_dataset.py` | `7eb3d18acde431d0` |
| `build_pw_solvers.py` | `1683a631a93918e0` |
| `cc_split.py` | `c2ad798e746ee7ae` |
| `clean_targets.py` | `e5e78fe96759d13e` |
| `clear_partial_targets.py` | `ab27d66a0ef3c8af` |
| `collect_panel_cells.py` | `c5df86f218ecb9f5` |
| `commit_split.py` | `5d2a0b0e1ffcee66` |
| `diag_pool_kl.py` | `d0d942626ad5c1ee` |
| `exp2_pipeline.py` | `1718fc8fc28f8abb` |
| `exp2_stage2.py` | `a4dc999d3f45c9fd` |
| `factorial_2x2.py` | `43ba0422a53598b2` |
| `filter_and_materialize.py` | `348f23b16c22117a` |
| `filter_prompt_level.py` | `d8fe2a6c6fd99db5` |
| `fix_cal.py` | `5ea513ace385bfe2` |
| `fix_dataset_env.py` | `8a9e95d089be2d91` |
| `fix_dev_shards.py` | `c5efea7295184223` |
| `fix_dpo_dataset.py` | `245d339abdb44755` |
| `fix_target_manifest.py` | `aa5dd64e0d92076a` |
| `freeze_safe_panel.py` | `ad9370a08aa1ea25` |
| `freeze_safe_splits.py` | `de968e095a163f39` |
| `freeze_uf_panel.py` | `79785c03f36e951f` |
| `freeze_wild_panel.py` | `03f4b905748beb5a` |
| `gate_uf_arms.py` | `8c57291dad0a83f3` |
| `gen_candidates.py` | `38d4ac85a7460659` |
| `generate_pros_pool.py` | `7792cdc6dee904c3` |
| `judge_eval_batch.py` | `72dadb73533bd1d3` |
| `judge_eval_batch_ahv2.py` | `3131fe76a233fc01` |
| `judge_eval_pairwise.py` | `9cbbfff767d21966` |
| `judge_fresh_safe.py` | `729e204f81a5f265` |
| `judge_pairs.py` | `cbdcbf7d5e65a635` |
| `judge_pool_items.py` | `34b6753c86f91866` |
| `judge_pool_pairs.py` | `dabe8d88f4c13e32` |
| `judge_pool_prosper.py` | `0cb23a8b90050bd0` |
| `judge_wild_items.py` | `567b5e71a1dce476` |
| `lc_winrate.py` | `e37049402ba47c3d` |
| `length_control.py` | `5b07c5ae9b88b1ae` |
| `make_pros_train_jobs.py` | `bcabadf496632acb` |
| `make_safe_solvers.py` | `7d757cf041f1e3c6` |
| `make_safe_solvers2.py` | `993e0e101de32a29` |
| `make_splits_and_queue.py` | `33422276d76041e4` |
| `make_train_v2.py` | `b2e65a54d032a0cc` |
| `merge_ext.py` | `f4a16ea8e9dd158e` |
| `mtbench_generate.py` | `5f57d2c8740e2494` |
| `mtbench_judge.py` | `1007bf7e9f74f674` |
| `mtbench_judge_batch.py` | `733864ed29ad2ed8` |
| `nbpo_local_gate.py` | `a5acfd5dbe44b483` |
| `nbpo_local_gate_selftest.py` | `5d96083801cf0306` |
| `paired_diff.py` | `8c02d7c6107a1add` |
| `paired_eval_diff.py` | `d63433fd17d62928` |
| `panel_eval_judge.py` | `43c34dc22a5fb653` |
| `panel_eval_matrix.py` | `1a0d6f3537474ac7` |
| `panel_eval_selftest.py` | `21e08fa423c944f8` |
| `panel_mopo.py` | `52cc67fd9c38b839` |
| `panel_solver_diag.py` | `7ac5b829389dc9f2` |
| `panel_stage2.py` | `e0528503bf22fa97` |
| `patch_prosper.py` | `83b304f6529266af` |
| `patch_solver_precision.py` | `876327e29824383f` |
| `probe_feasible.py` | `1723afa0a22fd8e0` |
| `probe_feasible_us1.py` | `3ea83b98426213d4` |
| `probe_feasible_ut1.py` | `3d1dded1cc3ff48e` |
| `probe_feasible_uw1.py` | `53beb08ef84e4cd9` |
| `probe_feasible_uw1c.py` | `31a0be3911903894` |
| `probe_feasible_uw3.py` | `acde1da41b4895dc` |
| `probe_surrogate_cost.py` | `af89f21cf1e1d840` |
| `prosper_targets.py` | `c5c56d17cd050e63` |
| `prosper_targets_panel.py` | `b12c3a216820c543` |
| `queue2.py` | `8010700db9e51882` |
| `queue_base_row.py` | `0d7ad2d55d88f51e` |
| `queue_controller.py` | `669509117674289f` |
| `queue_dpo.py` | `9c91388d26df16f1` |
| `queue_fresh_arms.py` | `468d46553b6ea1ea` |
| `queue_label.py` | `df26873825501752` |
| `queue_safe_screen.py` | `58fceff3c7cc3ca8` |
| `queue_solve_train.py` | `4b464ec97456008c` |
| `queue_uf_screen.py` | `4ca4719b7638f0f9` |
| `queue_wild.py` | `a8aa7fa719093b97` |
| `record_and_train.py` | `f428f47382634dc4` |
| `redispatch4.py` | `0ca3eca20f5098a5` |
| `redispatch5.py` | `1e095684e54db881` |
| `regen_configs.py` | `5ea90bc8b943cba7` |
| `repoint_cc.py` | `f19931f7042dcbe2` |
| `repoint_solvers.py` | `6804b8c1c54c324a` |
| `requeue_solves.py` | `9f4a65a1c9641638` |
| `restrict_common_certified.py` | `193c4e4e01aa7bf1` |
| `restrict_v2.py` | `b876b7b9118e4c94` |
| `score_pros_pool.py` | `13f9dcb7dfe96f23` |
| `score_safe_pool.py` | `51b0bdea4a8069ad` |
| `setup_panel_eval.py` | `bb1b4f30b05cdf33` |
| `solve_mopo_targets_us1.py` | `29ffdad4c60edf68` |
| `solve_mopo_targets_ut1.py` | `d4d88c0f3240f98e` |
| `solve_pros.py` | `6437b4763d8bae50` |
| `solve_pros4_prosper.py` | `4acef1091e95e5d1` |
| `solve_pros4_prosper_us1.py` | `a424790acb05c120` |
| `solve_pros4_prosper_ut1.py` | `f3f164f01e67691f` |
| `solve_pros4_prosper_uw1.py` | `9bc12258adb0f362` |
| `solve_pros4_prosper_uw1c.py` | `24813e435311a09d` |
| `solve_pros4_prosper_uw3.py` | `882ed6e6698420e4` |
| `solve_pros4_prosper_v2.py` | `e5788f0897387777` |
| `solve_pros4_pw_fixedref.py` | `7a9f2b4fc4820b7b` |
| `solve_pros4_pw_fixedref_us1.py` | `f28d388b740106b6` |
| `solve_pros4_pw_fixedref_ut1.py` | `93db4e6c6443b50e` |
| `solve_pros4_pw_fixedref_uw1.py` | `154bdf73323a8782` |
| `solve_pros4_pw_fixedref_uw1c.py` | `3ab9c86049b3617f` |
| `solve_pros4_pw_fixedref_uw3.py` | `96f091b8084e43cb` |
| `solve_pros4_pw_fixedref_v2.py` | `e3fdc6beaa2a352c` |
| `solve_pros4_pw_nbpo.py` | `1d9b0953162ca88f` |
| `solve_pros4_pw_nbpo_us1.py` | `2ff143d2361e8492` |
| `solve_pros4_pw_nbpo_ut1.py` | `459eabc33fd2a3cb` |
| `solve_pros4_pw_nbpo_uw1.py` | `d9b0fa546f1660d1` |
| `solve_pros4_pw_nbpo_uw1c.py` | `5d7f022f99f3f300` |
| `solve_pros4_pw_nbpo_uw3.py` | `95b3090e0b6be64c` |
| `solve_pros4_pw_nbpo_v2.py` | `575eeff8d0f9a9a3` |
| `solve_pros4_targets.py` | `dc9a8ac57dbbece5` |
| `solve_pros4_targets_us1.py` | `001d378f0ddf3cd7` |
| `solve_pros4_targets_ut1.py` | `dc71344084bdfa51` |
| `solve_pros4_targets_uw1.py` | `43ad471fa90a19c5` |
| `solve_pros4_targets_uw1c.py` | `f392ff5452639e5a` |
| `solve_pros4_targets_uw3.py` | `a524b3d7e4f3774c` |
| `solve_pros4_targets_v2.py` | `dc9a8ac57dbbece5` |
| `solve_safe_prosper.py` | `b7f23de80f1d5a45` |
| `solve_safe_targets.py` | `74e3181a39074b3c` |
| `sweep_diag.py` | `f18de8a6a82291a3` |
| `sweep_l1.py` | `bcbb8c5d65bd21cf` |
| `terminal_prosper.py` | `4d747c4540927d46` |
| `unify_scoring.py` | `1788095d71d88995` |
| `union_freeze_us.py` | `df2cd2f1b80cb441` |
| `union_freeze_ut.py` | `6e13bb4739c29d76` |
| `union_freeze_uw.py` | `9b615f97c2e76f65` |
| `union_judge_uw.py` | `1e58249f87d6cc9d` |
| `union_score_panel.py` | `e3bd9c1c2486715f` |
| `union_score_uw.py` | `3752827f4cd0fcda` |
| `upload_e2_arms.py` | `83a23801b40a64d1` |
| `upload_uw_arms.py` | `a470d2c7f816d28e` |
| `uw_solver_diag.py` | `bc9079f9b717e9d2` |
| `wild_selftest.py` | `50984f97c019e949` |
| `xplay_judge.py` | `9e3ec950b76ccc28` |
| `xplay_matrix.py` | `5ba2c58e3dc58b2f` |
| `eval_panel_ahv2.json` | `f411aebfadbfa1a8` |
