# UF-4 round code snapshot, 2026-09-17

Byte-identical copy of `/work/uf4_20260910/code` on the campaign pod. Every
file here hashes the same as its source at the time of the copy; the sha256
prefix is recorded per file in the table at the end.

This is the four-objective UltraFeedback round: the teacher fits, candidate
pool, solvers, training queue, capability benchmarks and audits behind the
manuscript's main comparison rows. Paths inside these scripts are the pod's
absolute paths (`/work/uf4_20260910/...`), so they are a record of what ran
rather than a portable package; the objective list is
`instruction_following, truthfulness, honesty, helpfulness`.

`queue_controller.py` also appears in
`../../sub_20260914/code_snapshot_20260917_union/`, at the same hash — one
controller serves both roots.

## Queue and scheduling

- `queue_controller.py` — the dependency queue: specs under
  `jobs/queue/<priority>_<job_id>.json`, taken in ascending
  `(priority, job_id)`; DONE never re-runs; FAILED or BLOCKED re-run only when
  `spec_sha256` changes; free devices are held for the top blocked 4-GPU job.
  `load_specs` skips a spec that vanishes mid-poll, because renaming one used
  to kill the controller with `FileNotFoundError`.
- `preflight.py` — asks of every queued 4-GPU training whether it would
  survive step 0, comparing the pinned dataset and solver hashes on disk first.
  MOPO had already failed twice at step zero on a hash mismatch, each time
  costing a four-GPU slot. Its own first version raised a false alarm on 10 of
  14 arms; the docstring records both mistakes it now avoids.
- `heartbeat.py` — appends the measured state of the campaign (live pids, GPU
  utilisation as the driver reports it, artifacts on disk) at a fixed interval.
- `make_train_jobs.py`, `make_pool_jobs.py`, `make_dpo_train_jobs.py`,
  `make_profile_jobs.py`, `queue_*.py`, `requeue_*.py`, `redispatch`-style
  helpers — per-arm configs and queue specs for pools, trainings, panels,
  cross-play, MT-Bench, capability benchmarks and rescoring.
- `note_s44_stop.py` — writes the operator's reason for stopping PROSPER seed
  44 next to that run's own log, so the record sits with the run.

## Data, teachers and targets

- `build_uf4_splits.py`, `build_uf4_splits_reproduce.py` — the prompt splits
  and an independent reproduction of them.
- `generate_uf4_pool.py`, `score_uf4_pool.py` — candidate pool generation and
  scoring.
- `train_uf4_teacher.py`, `show_teachers.py`, `test_uf4_teacher.py` — the
  per-objective teacher fits (Bradley-Terry and GPM) with their NLL, Brier and
  tie diagnostics, and the selected-step report.
- `solve_uf4_targets.py`, `solve_prosper_targets.py`, `solve_mopo_targets.py` —
  Nash, absolute max-min and MOPO targets for this round.
- `build_pm_pairs.py`, `build_dpo_pairs.py`, `build_dpo_scalarized.py`,
  `build_mopo_rows.py`, `materialize_dpo_dataset.py`,
  `materialize_mopo_dataset.py`, `rematerialize_dataset.py`,
  `fix_manifest.py` — baseline dataset construction and manifests.
- `build_eval_prompt_bank.py`, `reserve_audit_prompts.py` — the evaluation
  prompt bank and the reserved audit prompts.
- `download_assets.py` — asset staging.

## MOPO's constraints

- `mopo_why.py` — measures, rather than asserts, whether raising the primary
  objective on this pool also raises the other three, through the within-prompt
  rank association across the eight candidates.
- `mopo_ratchet.py` — simulates the paper's ratchet (`b = beta * G(rho)` raised
  every `t0` steps) with array arithmetic only, and reports whether the dual
  leaves zero. With `t0 = infinity` the measured dual was `lambda = 0`, so the
  constraints are vacuous at our horizon; the simulation is what decides
  whether that is an artefact of the horizon or the mechanism.

## Evaluation

- `judge_final_eval.py`, `paired_final_eval.py`, `aggregate_common_final.py` —
  final pairwise evaluation, the paired bootstrap, and aggregation over the
  arms' common prompts.
- `judge_crossplay_pair.py`, `aggregate_crossplay.py`,
  `requeue_crossplay.py` — arm-versus-arm cross-play.
- `generate_mtbench.py`, `score_mtbench.py`, `test_score_mtbench.py`,
  `queue_mtbench.py` — MT-Bench generation and grading with its parser test.
- `run_lm_eval.py`, `run_lm_eval_uw.py`, `collect_capability.py`,
  `collect_base_metrics.py`, `collect_deterministic_bench.py`,
  `queue_capability_gap.py`, `queue_deterministic_bench.py` — the capability
  table.
- `score_xstest_local.py`, `test_xstest_parse.py` — XSTest scoring and parser
  test.
- `eval_nbpo_trend.py`, `train_trend.py`, `add_stage_cis.py` — the stage trend
  and its intervals.
- `select_competitor.py` — applies the competitor-selection rule as declared on
  2026-09-12, before any of it was measured: highest minimum-objective win rate
  on `policy_dev` under the frozen independent judge, ties beyond 0.002 broken
  by average then by name. It refuses to read `evaluation/final_eval` for a
  non-devsel arm, because those are the outcomes the contract forbids the
  decision from seeing.
- `add_bounds.py` — the contract's missing-data bounds: every unresolved
  preference set to 0 and to 1, reported on all 200 planned prompts instead of
  only the complete cases.
- `score_summary.py`, `collect_costs.py`, `collect_cost_table.py`,
  `profile_report.py` — score and cost summaries.

## Audits and diagnostics

- `audit_uf4_solver.py`, `audit_coverage_quality.py`, `analyze_audit.py`,
  `run_audit_judge.py`, `generate_audit_responses.py`, `select_audit_witness.py`,
  `inspect_witness.py`, `test_audit_analysis.py` — the solver and coverage
  audits with their witness selection.
- `diag_conflict.py`, `diag_target_signal.py`, `diag_neural_pools.py`,
  `diag_tensor_contract.py`, `diag_loss_identity.py`, `diag_manifest.py`,
  `diag_select_panel.py`, `diag_aggregate_bank.py`, `diag_judge_bank.py`,
  `diag_judge_fresh.py`, `diag_gen_fresh.py`, `diag_build_pilot_dataset.py`,
  `diag_pilot_table.py`, `diag_pilot_fresh_ci.py`, `queue_diag_*.py`,
  `extend_logratio_job.py` — objective conflict, target signal, tensor and loss
  identity checks, and the projection pilot.
- `diag_think.py` — tests whether the 256-token cross-play parse collapse was
  Qwen3 thinking-mode overrun, by checking whether failing outputs stopped
  inside an unclosed reasoning block.
- `patch_save_deltas.py`, `patch_sequential_sampler.py` — the two trainer
  patches this round applied.

## Not included

Pools, judgments, teacher checkpoints, solver artifacts, datasets, trained
checkpoints and evaluation outputs stay on the pod; only code is snapshotted.

## File hashes

| File | sha256 (16) |
|---|---|
| `add_bounds.py` | `baeb4fd041f85b74` |
| `add_stage_cis.py` | `56dd21aafaf8ecf7` |
| `aggregate_common_final.py` | `9b3fa7f9e67a86c2` |
| `aggregate_crossplay.py` | `f6f388728f82e41e` |
| `analyze_audit.py` | `f2920f1a880f525f` |
| `audit_coverage_quality.py` | `747d9a75ddac3d68` |
| `audit_uf4_solver.py` | `0f8b1f557a9555f3` |
| `build_dpo_pairs.py` | `e991971f6ab0c313` |
| `build_dpo_scalarized.py` | `ca0bc544e0f885b7` |
| `build_eval_prompt_bank.py` | `f21777c8a778a1b9` |
| `build_mopo_rows.py` | `da070866e88d649a` |
| `build_pm_pairs.py` | `8cd426c964f1e98f` |
| `build_uf4_splits.py` | `c11744ea808c01e8` |
| `build_uf4_splits_reproduce.py` | `197c887749d62898` |
| `collect_base_metrics.py` | `7ae43e30029e2534` |
| `collect_capability.py` | `96962f2d58ba610b` |
| `collect_costs.py` | `f02d1631f44a4cba` |
| `collect_cost_table.py` | `3db2612e05d770f5` |
| `collect_deterministic_bench.py` | `d21eb591f38691cd` |
| `diag_aggregate_bank.py` | `b5bba3c5f8403fdc` |
| `diag_build_pilot_dataset.py` | `dd9f941fd2b13270` |
| `diag_conflict.py` | `e01a92a31db5d87a` |
| `diag_gen_fresh.py` | `bd6615249ee52324` |
| `diag_judge_bank.py` | `6c9c07f149102506` |
| `diag_judge_fresh.py` | `02590ef1bf2014aa` |
| `diag_loss_identity.py` | `b909c82f19cec00a` |
| `diag_manifest.py` | `8b900da8967617dc` |
| `diag_neural_pools.py` | `2e5b06529cd8f4c3` |
| `diag_pilot_fresh_ci.py` | `358d42b5c063fa9b` |
| `diag_pilot_table.py` | `535a0386da9639ab` |
| `diag_select_panel.py` | `aa543c2317314846` |
| `diag_target_signal.py` | `d1745e0f4a2db2a1` |
| `diag_tensor_contract.py` | `55ca061d19784b4f` |
| `diag_think.py` | `c975ff51c545def5` |
| `download_assets.py` | `80ade52e7e187d31` |
| `eval_nbpo_trend.py` | `cefae39649b9e2cb` |
| `extend_logratio_job.py` | `ec483c54c1206b29` |
| `fix_manifest.py` | `d11a134a061a34ec` |
| `generate_audit_responses.py` | `a5a8c98b32be78e5` |
| `generate_mtbench.py` | `b496a22be2dd7b83` |
| `generate_uf4_pool.py` | `eadb32374e8bfa1d` |
| `heartbeat.py` | `5fe41df57aedf80b` |
| `inspect_witness.py` | `30d61906e87be3e2` |
| `judge_crossplay_pair.py` | `c8697fb9ccad7771` |
| `judge_final_eval.py` | `c32167229b06d6b5` |
| `make_dpo_train_jobs.py` | `e3360636a6742097` |
| `make_pool_jobs.py` | `4b02d9d5848846ee` |
| `make_profile_jobs.py` | `3a3568f4ab2bd3af` |
| `make_train_jobs.py` | `e796a5d00000bd59` |
| `materialize_dpo_dataset.py` | `6bc6ee3268eae27f` |
| `materialize_mopo_dataset.py` | `e0fa4b7f0f746ff3` |
| `mopo_ratchet.py` | `209912c13cfac16a` |
| `mopo_why.py` | `8119959548c91c88` |
| `note_s44_stop.py` | `34fdd8ef96edd498` |
| `paired_final_eval.py` | `196496dc5473c699` |
| `patch_save_deltas.py` | `8b38117b4ae82ece` |
| `patch_sequential_sampler.py` | `6fb80084a398c939` |
| `preflight.py` | `490e459bd7b97f6c` |
| `profile_report.py` | `1e6d03f43d045d44` |
| `queue_arm_finaleval.py` | `b1105b76bd1657a8` |
| `queue_arm_panel.py` | `914729f8bbb2cc58` |
| `queue_base_capability.py` | `028f9077cd2ddf4a` |
| `queue_base_fresh.py` | `cb652a6e24108cf1` |
| `queue_capability_gap.py` | `a0406fbb94e9abc6` |
| `queue_controller.py` | `669509117674289f` |
| `queue_deterministic_bench.py` | `317b861dc6f6d3e9` |
| `queue_diag_fresh.py` | `705cc4d52c18a42b` |
| `queue_diag_jobs.py` | `88659b552085ecc7` |
| `queue_diag_projection.py` | `03dc32d9a9906b8e` |
| `queue_final_base.py` | `2bdf7a88aad31a79` |
| `queue_mtbench.py` | `6491983599585d27` |
| `queue_pilot_followups.py` | `1899a1374d6fde49` |
| `queue_rescore_21arm.py` | `62fbd5255a09ebf1` |
| `queue_rescore.py` | `d0260ddb0caf5a07` |
| `queue_rescore_s44.py` | `fb846780cbf93f04` |
| `queue_solve_chain.py` | `90190e3d226c0c0a` |
| `rematerialize_dataset.py` | `4f6b674fe61feb0f` |
| `requeue_crossplay.py` | `a8a46d3c334651e3` |
| `requeue_s44.py` | `80a1f216a10c8b6f` |
| `reserve_audit_prompts.py` | `5d806a55f904f154` |
| `run_audit_judge.py` | `512fb2c3922f4439` |
| `run_lm_eval.py` | `ba589eab91a89a24` |
| `run_lm_eval_uw.py` | `f13f58965f052a66` |
| `score_mtbench.py` | `da4f0e3d6262cb9e` |
| `score_summary.py` | `52beff6b72902a47` |
| `score_uf4_pool.py` | `b91dc73a5aa37fae` |
| `score_xstest_local.py` | `e2a747c0f235330b` |
| `select_audit_witness.py` | `09ed56ed0320c1ab` |
| `select_competitor.py` | `d70d1186c15a91f5` |
| `show_teachers.py` | `69718af59431fa17` |
| `solve_mopo_targets.py` | `5a61412b1de7a65f` |
| `solve_prosper_targets.py` | `1809a5f5b63019c3` |
| `solve_uf4_targets.py` | `2ff97b1b496099c8` |
| `test_audit_analysis.py` | `5a089b531aa09792` |
| `test_score_mtbench.py` | `96213205375d1eaf` |
| `test_uf4_teacher.py` | `8477ba48cb6b0d94` |
| `test_xstest_parse.py` | `3699f7e720f294a4` |
| `train_trend.py` | `56643ce6f73c9b85` |
| `train_uf4_teacher.py` | `b5fcc6ac0fab2005` |
