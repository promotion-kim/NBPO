# Codex task — Open the Qwen3-8B sealed reward test and produce the headline robustness table (deadline 22:00 KST, 2026-07-14)

You are operating in the `MNPO` repository on the B200 machine. Your job is the single highest-value experiment remaining for the RONPO AAAI paper: finalize model selection on the already-frozen validation split, then open the held-out **sealed reward test exactly once** and score every flagship model on it, producing the paper's headline worst-objective robustness numbers. Finish and leave reviewable results by **22:00 KST tonight**.

## Non-negotiable constraints (read first)

1. **The sealed test is opened once and never used for selection.** Selection must be finalized on the non-sealed validation split before any sealed generation or scoring runs. After you open it, do not re-tune, re-select, or re-open based on sealed numbers. If you touch the sealed prompts before selection is locked, the result is void.
2. **No fabrication.** Every number comes from a measured artifact. Never impute, mirror, or hand-edit a score. If a step fails, record it and stop that branch; do not synthesize a value.
3. **Fail-closed stability gate.** A model that fails the generation stability gate is reported as failed, not silently dropped or patched.
4. **GPU etiquette.** Use only the B200 GPUs authorized for this account. Take three read-only `nvidia-smi` samples first and confirm the target GPUs are idle and owned by no other user. Never kill, pause, or attach to another user's process. If a GPU is busy with someone else's job, leave it alone.
5. **Fair protocol.** The sealed evaluation must reuse the exact decode and scoring configuration that produced the validation split (`results/p1_validation_reward_seed42_20260714/`), so validation and sealed are comparable. Do not change decode temperature, top-p, max tokens, thinking flag, reward heads, normalization, or bootstrap settings.
6. **Deadline.** Do not launch any new job that cannot finish and be verified by 22:00 KST. Prefer completing the sealed table for the already-eligible models over waiting on the sweep.

## Background (already established, do not redo)

- The 11 flagship seed-42 models are trained, stability-gated, and public. Their names, HF repos, and exact revisions are in `results/qwen3_seed42_academic_20260714/final/models.tsv` (base plus dpo, ipo, simpo, sppo_avg, inpo_avg, ht_mnpo_{helpfulness,safety,conciseness}, and the two RONPO estimators `ronpo_k_only` = top-mass and `ronpo_full_expect` = full-expectation).
- On the 128-prompt non-sealed validation split (`results/p1_validation_reward_seed42_20260714/ranked_validation_summary.json`), the frozen primary metric `mean_prompt_worst_norm_score` ranks `ronpo_k_only` (top-mass) first at 0.2853, above base (0.2587) and every baseline. `ronpo_full_expect` is eighth.
- A P3 sweep (`results/p3_ronpo_seed42_sweep_protocol_v2_20260714.json`, status in `results/p1_validation_reward_seed42_20260714/p3_status.json`) is retraining full-expectation candidates. As of the last status it had produced no eligible candidate.
- The three objectives are helpfulness, safety, and conciseness scored by ArmoRM heads. The conflict gate that certified they are genuinely opposed is `results/ronpo_flagship_20260712/conflict_gate_primary.json`.
- The sealed prompts (604 prompts) are defined by the frozen split manifest; a verified local copy is `results/ronpo_flagship_resume_20260713/sealed_test_prompts.jsonl` (SHA-256 `52b4028bd3ce095524e3ae66f49bf495d1236fea4635248b4263f9db1920df69`).

## Phase A — Lock selection (validation only, no sealed access)

1. Reload `results/p1_validation_reward_seed42_20260714/ranked_validation_summary.json` and confirm `p1_sealed_test_opened == false`.
2. Check `p3_status.json`. For each sweep candidate that has (a) finished its 900 training steps, (b) passed the frozen S3 stability gate, and (c) been scored on the 128-prompt validation split with the same 3-head pipeline: include its validation `mean_prompt_worst_norm_score`.
3. Apply the frozen selection rule (`.../ranked_validation_summary.json` metric plus the p3 protocol's rule): the selected RONPO entry is the eligible candidate with the highest validation worst-objective score, breaking ties by IFEval. If no sweep candidate exceeds the current top-mass 0.2853 and can be validation-scored by 19:00 KST, select `ronpo_k_only` (top-mass). Do not block on the sweep.
4. Write `results/p1_sealed_reward_seed42_20260714/selection_lock.json` recording the selected RONPO variant, the validation ranking used, and a statement that no sealed data was consulted. This file is the GO signal for Phase B.

## Phase B — Open the sealed test once

Reuse the exact runner that produced the validation reward directory (find it; it is the script referenced by that run's status/wandb metadata, and it wraps vLLM decode, `on_policy_data_gen/rm_armo.py` scoring, and `mnpo_scripts/evaluate_multi_objective_models.py` aggregation). Point it at the sealed split instead of validation. Score **all 11 flagship models plus the selected RONPO** (score both RONPO estimators so the ablation row is available).

Frozen configuration (must match validation):
- Decode: vLLM, seed 42, temperature 0.7, top-p 0.9, max_new_tokens 2048, chat template with thinking disabled, bfloat16, exact HF revisions from `models.tsv`.
- Scoring: ArmoRM helpfulness, safety, conciseness heads per `conflict_gate_primary.json`.
- Normalization: per-prompt min–max across the evaluated sealed pool; per-prompt average and worst objective; win rate vs the base response.
- Intervals: 2000-resample paired prompt-level bootstrap, seed 42.
- Stability gate (fail-closed, per model over the 604 sealed generations): 604 records, 0 empty, 0 think-tag leakage, mean-word ratio vs base in [0.33, 2.0], max consecutive identical word run <= 20.

Maximize B200 utilization: shard the 604 prompts by 11 models across all authorized GPUs, run multiple vLLM engines in parallel, and keep RM scoring pipelined behind decode. Log progress hourly.

## Phase C — Artifacts and paper update

1. Write to `results/p1_sealed_reward_seed42_20260714/results/`:
   - `ranked_sealed_summary.json` (same schema as the validation `ranked_validation_summary.json`, sorted by `mean_prompt_worst_norm_score`, with bootstrap CIs and win rates),
   - `per_objective_scores.csv` (same columns as the validation file),
   - `SEALED_REPORT.md`: a human-readable ranked table (worst, avg, win rate, per-objective, stability), the selected RONPO variant, and a provenance block (decode config, RM revisions, prompt count, SHA-256 of the sealed file, wandb run ids). Set `p1_sealed_test_opened = true` in the status file.
2. Regenerate the paper table and figure from the sealed numbers (the build script auto-detects the sealed file):
   ```
   python3 analysis/ronpo_8b_reconstruction_20260714/build_ronpo_8b_tables_figure.py \
       --reward-dir results/p1_sealed_reward_seed42_20260714
   ```
3. In `ronpo_aaai/main.tex`, replace the body of the `tab:qwen3-robust-validation` table with the regenerated `analysis/ronpo_8b_reconstruction_20260714/table_main_reward_robustness.tex`, copy the regenerated `fig_worst_objective.pdf` to `ronpo_aaai/figures/qwen3_worst_objective.pdf`, and change the two phrases "validation selection split" in the surrounding paragraph and figure caption to "sealed test". Do not change any other paper content. Recompile with `PATH=/home/sjkim/MNPO/.TinyTeX/bin/x86_64-linux:$PATH pdflatex ... ; bibtex ; pdflatex x2` and confirm 0 fatal errors and 0 overfull hboxes on the changed pages.

## Acceptance criteria (report these back)

- `selection_lock.json` written before any sealed generation, naming the selected RONPO variant.
- `ranked_sealed_summary.json` and `per_objective_scores.csv` present with all evaluated models, finite numbers, and CIs.
- Sealed worst-objective rank of RONPO (top-mass), with its CI, stated plainly, whether or not it is rank 1. Report the true result either way.
- Stability gate outcome per model.
- `main.tex` updated and recompiling clean.
- A short provenance summary and the final ranked sealed table pasted into your closing message so the result can be reviewed at a glance.

If anything blocks completion by 22:00 KST, stop cleanly, write what was and was not measured, and leave the sealed test unopened rather than opening it under a rushed or non-frozen configuration.
