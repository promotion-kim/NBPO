# Codex task — Resume and complete the already-opened Qwen3-8B sealed reward evaluation, then fill spare time with the highest-value AAAI'27 experiments (hard deadline 09:00 KST, 2026-07-15)

You are operating in the `MNPO` repository on the B200 machine. The Qwen3-8B sealed reward test was **already opened once** earlier today and then stopped by a fail-closed stability gate before any reward scoring ran. Your job is to **complete that single, already-authorized measurement by resuming from the responses that were already generated** — no new decode, no re-selection, no second opening — produce the paper's headline sealed worst-objective robustness numbers, land them in the paper, and then use the remaining time before 09:00 KST to run the highest-value follow-on experiment for the AAAI'27 main-track case. Leave reviewable results by **09:00 KST on 2026-07-15**.

Read this whole document before touching anything.

## What already happened (established facts — do not redo, do not second-guess)

- Selection was locked on the non-sealed validation split at `2026-07-14T10:23:51+09:00`. The selected RONPO variant is `ronpo_k_only` (top-mass). Evidence: `results/p1_sealed_reward_seed42_20260714/selection_lock.json` and `.../sealed_opened.json`. **Selection is final. Do not re-select, re-tune, or re-rank on anything.**
- The 604-prompt sealed test was opened once at `2026-07-14T10:24:47+09:00`, sealed prompts SHA-256 `52b4028bd3ce095524e3ae66f49bf495d1236fea4635248b4263f9db1920df69` (local copy `results/ronpo_flagship_resume_20260713/sealed_test_prompts.jsonl`).
- All 11 flagship models **already generated exactly 604 non-empty responses each**. They are on local disk at `results/p1_sealed_reward_seed42_20260714/generations/<model>/output_42.json` (with `decode_metadata.json`). Models: `base`, `ronpo_k_only`, `ronpo_full_expect`, `dpo`, `ipo`, `simpo`, `sppo_avg`, `inpo_avg`, `ht_mnpo_helpfulness`, `ht_mnpo_safety`, `ht_mnpo_conciseness`. HF repos/revisions: `results/qwen3_seed42_academic_20260714/final/models.tsv`.
- Frozen decode config (already applied, in each `decode_metadata.json`): vLLM, seed 42, temperature 0.7, top-p 0.9, max_new_tokens 2048, bf16, chat template with thinking disabled.
- The run stopped at stage `sealed_stability_gates` with every model marked failed (`results/p1_sealed_reward_seed42_20260714/status.json`, `.../stability_gates/summary.json`, `.../results/SEALED_REPORT.md`). **No reward number was ever computed and the paper was correctly left unchanged.**

### Root cause of the gate failure (already diagnosed — verify, then act on it)

The gate is `scripts/revision/flagship/stability_gate.py`. Its think-leak check flags a record if `generated_text_raw` contains any `<think>` or `</think>` substring. Two distinct things tripped it:

1. **Ten of eleven models fail only because of a single record at sealed prompt index 394.** That prompt text ends with `"OK. Let's think. My stream of consciousness:"`, an adversarial coax. In Qwen3 non-thinking mode the chat template renders an empty `<think></think>` scaffold, and at this one prompt the model emits a stray orphan `</think>` (no opening tag, no hidden reasoning; the answer after it is a normal response). This is exactly 1 of 604 records and it appears in `base` too. This is a template/gate artifact, not a hidden-reasoning leak. The only failing check for these ten models is `zero_think_leakage`; length, repetition, empties, and record count all pass.
2. **`dpo` fails for a different, genuine reason:** a 1163-word verbatim repeat loop at index 252 (real degeneration; `max_repeat_run` far above the threshold of 20). This is a legitimate fail-closed outcome. **Do not patch, re-decode, or rescue `dpo`. Report it as a genuine stability failure.**

## Non-negotiable constraints (read first)

1. **One-shot sealed integrity is preserved by construction.** You are finishing the one authorized opening, not opening a second time. **Do not decode, regenerate, or sample the sealed prompts again. Do not re-run selection. Reuse the exact 604 generations already on disk.** The only thing you correct is the stability-gate detector, and you correct it by inspecting generation *text and tags*, never a reward score. Finalize the corrected gate rule and the pass/fail set **before** computing or looking at any reward number.
2. **No fabrication.** Every number comes from a measured artifact. If a step fails, record it and stop that branch. Never impute, mirror, or hand-edit a score.
3. **Fail-closed stays fail-closed for genuine failures.** A model that fails the corrected gate for a real reason (e.g. `dpo`'s repeat loop) is reported as failed, not scored, not silently dropped.
4. **GPU etiquette.** Take three read-only `nvidia-smi` samples before any launch and confirm the target B200 GPUs are idle and owned by no other user. Use only the B200 GPUs authorized for this account. Never kill, pause, or attach to another user's process.
5. **Fair protocol.** Reward scoring must reuse the exact scoring, normalization, and bootstrap configuration that produced the validation reward directory (`results/p1_validation_reward_seed42_20260714/`), so validation and sealed remain comparable. Do not change reward heads, normalization, or bootstrap settings.
6. **Deadline 09:00 KST 2026-07-15.** Do not start any job that cannot finish and be verified by then. Landing the sealed table for the gate-passing models takes priority over every spare-time experiment.

## Phase A — Correct the stability gate (outcome-blind), then re-run it on the existing generations

1. Implement a corrected think-leak rule that distinguishes a genuine reasoning leak from the template/orphan artifact. Count a leak only when there is a **non-empty `<think> ... </think>` reasoning span** (an opening `<think>` followed by non-whitespace content followed by a closing `</think>`). Treat a lone/orphan `</think>`, a lone `<think>`, and an empty `<think></think>` pair as template artifacts, not leaks. Apply the identical rule to all 11 models. Keep every other threshold byte-for-byte identical: exactly 604 records, zero empty, mean-word ratio vs base in [0.33, 2.0], max consecutive identical word run ≤ 20.
2. Write the corrected detector as a **new** file (e.g. `scripts/revision/flagship/stability_gate_corrected.py`) or a clearly-flagged option; **do not overwrite** the original `stability_gate.py` or the original failed gate JSONs. Preserve the original `stability_gates/` outputs; write corrected results alongside (e.g. `stability_gates_corrected/`).
3. Re-run the corrected gate over the existing generations. Record the before/after check table per model. Expected outcome to verify (report the true result regardless): `base`, `ronpo_k_only`, `ronpo_full_expect`, `ipo`, `simpo`, `sppo_avg`, `inpo_avg`, `ht_mnpo_helpfulness`, `ht_mnpo_safety`, `ht_mnpo_conciseness` pass; `dpo` still fails on the index-252 repeat loop.
4. Write `results/p1_sealed_reward_seed42_20260714/gate_correction.json`: the exact old rule, the new rule, the per-model before/after, a one-line statement that the correction was decided by inspecting generation text and tags with **no reward score consulted**, and the index-394 / index-252 evidence. This file is the audit trail and the GO signal for Phase B.

## Phase B — Score the already-generated sealed responses (no decode)

1. Find the scoring and aggregation stages of the sealed/validation runner (`scripts/revision/flagship/run_seed42_sealed_reward_eval.py`; the reward step is `score_armo_primary_heads.py` / `score_armo_primary_heads_sharded.py`, followed by the multi-objective aggregation the validation run used). Invoke **only** the scoring, normalization, bootstrap, and ranking stages, pointed at the existing `generations/<model>/output_42.json`. **Do not invoke the decode stage.**
2. Score every model that **passes the corrected gate**, including both RONPO estimators (`ronpo_k_only` and `ronpo_full_expect`) so the ablation row is available. `dpo` is excluded as a stability failure and reported as such, not scored.
3. Reward config, matching validation exactly: ArmoRM helpfulness, safety, conciseness heads per `results/ronpo_flagship_20260712/conflict_gate_primary.json`; per-prompt min–max normalization across the evaluated sealed pool; per-prompt average and worst objective; win rate vs the base response; 2000-resample paired prompt-level bootstrap, seed 42.
4. Maximize B200 utilization: shard the reward scoring across all authorized GPUs and run heads/models in parallel. Scoring is far cheaper than decode, so this stage should complete quickly; keep the pipeline saturated.

## Phase C — Artifacts and paper update (edit `main_v2.tex`, not `main.tex`)

1. Write to `results/p1_sealed_reward_seed42_20260714/results/`:
   - `ranked_sealed_summary.json` (same schema as the validation `ranked_validation_summary.json`, sorted by `mean_prompt_worst_norm_score`, with bootstrap CIs and win rates),
   - `per_objective_scores.csv` (same columns as the validation file),
   - update `SEALED_REPORT.md` to the completed (not failed) form: a human-readable ranked table (worst, avg, win rate, per-objective, stability), the selected RONPO variant (`ronpo_k_only`), the `dpo` gate failure stated plainly, and a provenance block (decode config, ArmoRM revisions, 604 prompt count, sealed SHA-256, reference to `gate_correction.json`, wandb ids). Set `status = completed` and keep `p1_sealed_test_opened = true`.
2. Regenerate the paper table and figure from the sealed numbers (the build script auto-detects `ranked_sealed_summary.json`):
   ```
   python3 analysis/ronpo_8b_reconstruction_20260714/build_ronpo_8b_tables_figure.py \
       --reward-dir results/p1_sealed_reward_seed42_20260714
   ```
3. In `ronpo_aaai/main_v2.tex`:
   - Replace the body of the `tab:qwen3-robust-validation` table with the regenerated `analysis/ronpo_8b_reconstruction_20260714/table_main_reward_robustness.tex`. **Keep the `\label{tab:qwen3-robust-validation}` unchanged** so no reference breaks.
   - Copy the regenerated `analysis/ronpo_8b_reconstruction_20260714/fig_worst_objective.pdf` to `ronpo_aaai/figures/qwen3_worst_objective.pdf`.
   - Update the paragraph beginning `\paragraph{Qwen3-8B scale-up: heterogeneous-objective robustness.}`, the `tab:qwen3-robust-validation` caption, the `fig:qwen3-worst` caption, and the Limitations sentence that currently says the 8B result "relies on a single training seed and a validation selection split rather than a sealed test," so that they describe the **sealed test** result. `grep` for `validation selection split` and for `validation` in the 8B-robustness context and update every occurrence that refers to this specific result, and only those.
   - **Report the true sealed ranking.** Do not assume the validation ordering carries over. State RONPO's sealed worst-objective rank and CI plainly whether or not it is rank 1, and soften or strengthen the surrounding claims to match the measured numbers. If RONPO is not first on the sealed worst objective, say so directly. Note the `dpo` stability failure honestly (a footnote or one appendix sentence is enough).
4. **Writing style for every new or edited sentence** (this matters):
   - Preserve all existing content. Make the smallest edits that carry the validation→sealed change and the corrected numbers. Do not restyle, reorder, or reword sentences that do not need to change.
   - Write new prose in clean, human, academic English. **Do not use em dashes (`---`)** or other AI-tell tics (no "delve", no "it's worth noting", no formulaic tricolons, no "Moreover/Furthermore" padding). Vary sentence length. Prefer plain connectives and direct phrasing. Read the added text back once and cut anything that sounds machine-generated. Match the register of the surrounding paper.
   - Prioritize readability: a reader should get the sealed result and its caveat in one pass.
5. Recompile and verify:
   ```
   cd ronpo_aaai
   PATH=/home/sjkim/MNPO/.TinyTeX/bin/x86_64-linux:$PATH pdflatex -interaction=nonstopmode -halt-on-error main_v2.tex
   PATH=/home/sjkim/MNPO/.TinyTeX/bin/x86_64-linux:$PATH bibtex main_v2
   PATH=/home/sjkim/MNPO/.TinyTeX/bin/x86_64-linux:$PATH pdflatex -interaction=nonstopmode -halt-on-error main_v2.tex
   PATH=/home/sjkim/MNPO/.TinyTeX/bin/x86_64-linux:$PATH pdflatex -interaction=nonstopmode -halt-on-error main_v2.tex
   ```
   Confirm 0 fatal errors, all references resolved, and no new overfull hboxes on the changed pages.

## Phase D — Spare-time experiments (only after A–C are fully landed and verified)

Because Phase B is score-only (the expensive decode is already done), A–C should finish in roughly one to three hours, leaving most of the night. Use it to strengthen the AAAI'27 main-track case, under two hard rules: **(i) never at the cost of the primary deliverable, and (ii) never re-open, re-decode, or re-select the sealed test.** Every Phase D item must fully complete and be verified by 09:00 KST or be left untouched with an honest note.

- **D1 (recommended first): reward-model-independent judge at 8B scale on the sealed generations.** The paper's ArmoRM heads overlap the training objectives, so an independent judge is the strongest missing signal. Reuse the existing local `gpt-oss-120b` arena pipeline (`scripts/revision/flagship/download_gptoss120_judge.py`, `judge_gpt4_anchor_gptoss.py`, `judge_gpt4_anchor_arena_verdict.py`, `aggregate_gpt4_anchor_gptoss.py`; prior run under `results/qwen3_gpt4_anchor_gptoss_eval_20260714/`). Run an all-pairwise judgment over the 604 sealed responses for the gate-passing models, ties counted as 0.5, deterministically randomized order, prompt-level bootstrap CIs. Lock the judging protocol in a small JSON before you judge. If it completes and verifies, add a compact result to the paper (appendix, same clean style, no em dashes) as an RM-independent sealed robustness check; otherwise write the artifacts and a note and leave the paper as of Phase C.
- **D2 (only if D1 is done and time remains): consistency refresh.** Re-run or verify the Qwen3-8B IFEval / academic-capability numbers on the sealed-eligible model set, or the openbench proxy judge (`run_openbench_proxy_pipeline.py`), for internal consistency. Lower value; optional.
- **Do not start any model training run overnight** (for example additional seeds). Full Qwen3-8B training cannot finish and be verified by 09:00 under this budget, and a partial run is worthless. If cross-seed robustness is desirable, only scope and prepare it (write the plan and the exact commands) and report readiness. Do not launch it.
- Maximize B200 utilization throughout: shard judging across all authorized GPUs, run engines in parallel, keep them busy.

## Monitoring and resume (every hour, until done or 09:00)

- Before any launch: three read-only `nvidia-smi` samples, confirm target B200s idle and unowned by others.
- Every hour, write a measured, rank-free status snapshot (reuse or adapt `scripts/revision/flagship/monitor_resume_hourly.py`) to `results/p1_sealed_reward_seed42_20260714/hourly/<timestamp>.json` with: current stage, GPU snapshot, per-model progress, and any errors seen.
- **If a step errors: diagnose it, fix it, and resume from the last good artifact.** Do not restart stages that already completed. Record every fix (what broke, what you changed) in the hourly snapshot and in a running `results/p1_sealed_reward_seed42_20260714/fix_log.md`. Never work around an error by fabricating or by silently dropping a model.

## Acceptance criteria (paste these into your closing message)

- `gate_correction.json` present, old and new rules stated, decided with no reward score consulted, original failed gate JSONs preserved.
- Corrected per-model gate outcomes, with `dpo` reported as a genuine stability failure (index 252, repeat run 1163).
- `ranked_sealed_summary.json` and `per_objective_scores.csv` present, all scored models, finite numbers, CIs.
- RONPO (top-mass) sealed worst-objective rank with its CI, stated plainly whether or not it is rank 1.
- `main_v2.tex` updated (validation→sealed, true numbers), recompiles clean, new prose free of em dashes and AI tells, existing content preserved, no new overfull hboxes.
- Phase D: exactly what completed and verified, plus an honest list of what did not and why.
- Hourly snapshots list and the `fix_log.md` contents.
- A short provenance summary and the final ranked sealed table pasted so the result can be reviewed at a glance.

## Stop rule

If anything blocks completion by 09:00 KST, stop cleanly, write what was and was not measured, and preserve integrity: never a second sealed opening, never a fabricated number, `dpo` stays a reported failure. Landing the sealed table for the gate-passing models is the one deliverable that must not be sacrificed to any spare-time experiment.
