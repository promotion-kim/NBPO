# Codex task — Localize the Qwen3-8B evaluation-power failure and, if possible, produce a real model-scale signal (hard deadline 09:00 KST, 2026-07-15)

You are operating in the `MNPO` repository on the B200 machine. The Qwen3-8B sealed reward evaluation completed honestly, but it has **no statistical power**: on the 604-prompt sealed test, all ten gate-passing models, base included, produce essentially identical ArmoRM-head rewards, so no method is separable and the reported worst-objective ranking is normalization noise. Your job tonight is to find out **why** and, if the data supports it, produce a genuine model-scale signal. Maximize B200 utilization. Leave reviewable results by **09:00 KST on 2026-07-15**.

Read this whole document before touching anything.

## Established facts (verify, then act; do not redo)

- The sealed test was opened once and scored. Selection (`ronpo_k_only`, top-mass) is locked. Artifacts: `results/p1_sealed_reward_seed42_20260714/results/{SEALED_REPORT.md,ranked_sealed_summary.json,per_objective_scores.csv}`, `.../gate_correction.json`.
- **The power failure, already measured:** in `.../results/per_objective_scores.csv`, the `mean_raw_score` of every model on each objective is within ~0.005 of every other model and of base (helpfulness 0.793–0.797, safety 0.874–0.879, conciseness −0.595 to −0.598). Per-prompt min–max normalization of these near-identical raw scores is what produced the 0.216–0.235 "worst-objective" spread with fully overlapping CIs.
- The policies did change: trained-model generations differ from base on roughly 77% of prompts (about 23% byte-identical, mean word Jaccard ~0.56). So the ArmoRM heads do not register the differences the training produced. This is head insensitivity / objective saturation, not identical policies.
- The 604 sealed generations are on disk at `results/p1_sealed_reward_seed42_20260714/generations/<model>/output_42.json` for the ten gate-passing models. DPO failed the gate (repeat loop at index 252) and stays failed.
- The validation split is `results/p1_validation_reward_seed42_20260714/` (128 prompts). The flagship models, HF repos, and revisions are in `results/qwen3_seed42_academic_20260714/final/models.tsv`. Reward scorers: `on_policy_data_gen/rm_skywork.py`, `rm_athene.py`, `rm_armo.py`. Training: `scripts/revision/flagship/train_flagship.py` and the flagship training configs.

## Non-negotiable constraints

1. **Do not re-open, re-decode, or re-select on the sealed test.** The sealed test is spent. Phase A re-scores the already-generated sealed responses with a different evaluator; that is a measurement of the evaluation, not a new selection. Never use any Phase A or Phase B number to re-pick the deployed model or to re-tune on sealed data.
2. **Pre-commit before you measure, to avoid fishing.** For Phase A, write the exact reward-model set to a locked JSON **before** scoring, and report every model's result regardless of whether RONPO looks good. Do not try several reward-model sets and keep the flattering one. One committed set, reported in full.
3. **No fabrication.** Every number comes from a measured artifact. A negative or null result is a result; report it plainly.
4. **Fail-closed stability gates** on any new generations, with the corrected think-leak rule from `gate_correction.json`. A model that fails a genuine check is reported as failed, not patched.
5. **GPU etiquette.** Three read-only `nvidia-smi` samples before any launch; only the authorized B200 GPUs; never touch another user's process.
6. **Deadline 09:00 KST.** Start nothing that cannot finish and verify by then. A finished honest diagnostic beats an unfinished retrain.
7. **Paper edits go in `ronpo_aaai/main_v2.tex`, never `main.tex`.** Preserve existing content, make minimal edits, no em dashes (`---`) and no AI-tell phrasing in new prose, prioritize readability, report true numbers.

## Phase A — Diagnostic re-scoring: are the 8B models separable at all? (fast, first, B200-sharded)

1. Pre-commit a reward-model set that is known to discriminate 1.5B policies in this repo and is stronger/independent of the ArmoRM heads: `Skywork/Skywork-Reward-V2-Llama-3.1-8B@cba2f842f3f1af2f1b2f0d35e794d789976390c5` and `Nexusflow/Athene-RM-8B@cdf428f7b52a323b6cf4e9803e5bcba9f1fb5a59`. Write this set, with revisions and the reason, to `results/p1_sealed_reward_seed42_20260714/power_diagnostic/rm_set_lock.json` before scoring.
2. Score the existing 604 sealed generations for all ten gate-passing models with each committed reward model. **No re-decode.** Shard across all authorized B200 GPUs to keep them busy.
3. For each reward model, report per-model mean raw score, the spread across models, per-prompt win rate vs base, and a 2,000-resample paired bootstrap CI (seed 42). The decisive question: does any committed reward model separate the trained models from base and from each other by a margin larger than its noise?
   - If the strong general reward models also show all models within noise, the Qwen3-8B policies are genuinely near-equivalent in reward, and the 8B section must be scoped as underpowered (Phase C).
   - If a committed reward model does separate them, record exactly which models move and by how much; this localizes the failure to the ArmoRM heads.
4. Write `results/p1_sealed_reward_seed42_20260714/power_diagnostic/REPORT.md` and `summary.json` with the full ranked results and the plain-language verdict.

## Phase B — Stronger-signal 8B retrain plus validation-split power check (B200-heavy main job)

The hypothesis is that the flagship training was too light for the policies to move enough to register on any judge. Test it directly.

1. Read the original flagship training config from `scripts/revision/flagship/train_flagship.py` and the flagship configs. Define a **substantially stronger** training configuration (for example more optimization steps, and/or a higher effective learning rate, and/or more UltraFeedback data or epochs), chosen so the policy is expected to move materially further from base than the flagship run did. Record the exact config in `results/p1_8b_retrain_stronger_20260715/config.json` with a one-line rationale.
2. Retrain, from the same base and seed 42, the models that matter for the thesis: both RONPO estimators (`ronpo_k_only` top-mass and `ronpo_full_expect`) and the two averaged-oracle baselines (`inpo_avg`, `sppo_avg`). Keep the same three opposed objectives (helpfulness, safety, conciseness) so the comparison is apples-to-apples. Parallelize the training runs across the four B200 GPUs; do not exceed what can finish and verify by 09:00. If four full runs do not fit, prioritize `ronpo_full_expect`, `ronpo_k_only`, and `inpo_avg`, and say what you dropped.
3. Apply the fail-closed stability gate to every retrained model.
4. Decode the gate-passing retrained models plus base on the **128-prompt validation split** (not the sealed test), with the frozen decode config (vLLM, seed 42, temperature 0.7, top-p 0.9, max_new_tokens 2048, bf16, thinking disabled). Score with the three ArmoRM heads and, additionally, the Phase A committed reward models.
5. **Raw-reward power check first.** Before computing any normalized ranking, verify that the retrained models now differ from base in raw reward by a margin clearly above noise on at least one objective. Only if they do is a normalized worst-objective ranking meaningful; report the raw spreads either way.
   - If the retrained models separate and RONPO holds the worst objective while an averaged baseline sacrifices one, that is a genuine model-scale signal on the validation split. Report it as a validation-split result, not a sealed one, and **pre-register a fresh sealed split** (new prompts, new SHA, written to a manifest) for a future single-shot confirmation. Do **not** open that fresh split tonight.
   - If the retrained models still do not separate, that is the honest finding: at this backbone and objective set, stronger training does not yield measurable heterogeneous-objective separation.
6. Write everything to `results/p1_8b_retrain_stronger_20260715/` (config, per-model gates, generations, scores, raw-spread power check, ranked validation summary with CIs, and a `REPORT.md`).

## Phase C — Paper update (honest, minimal, `main_v2.tex`)

1. **Regardless of Phase A/B outcomes**, correct the 8B robustness paragraph and caption to state the power finding accurately: the ten models produce statistically indistinguishable per-objective raw rewards (within ~0.005), so at this backbone and objective set the evaluation does not separate methods, and the reported ranking should not be read as a robustness result. This is more accurate than "the intervals overlap" and is the honest framing. Keep the table and figure; adjust the surrounding sentences and the caption only.
2. If Phase A shows a committed reward model separates the models, add a short, honest sentence or appendix note with that result and what it implies (ArmoRM-head saturation), reusing the JSON/CSV-driven build path so no number is hand-typed.
3. If Phase B produces a genuine validation-split signal, add it as a clearly-labeled validation-split result (not sealed), with the raw-reward power check stated, and note the pre-registered fresh sealed split as future work. If Phase B is null, add one honest sentence that stronger training did not change the separation, and leave the claim scoped.
4. Recompile with TinyTeX (`PATH=/home/sjkim/MNPO/.TinyTeX/bin/x86_64-linux:$PATH pdflatex main_v2 ; bibtex main_v2 ; pdflatex x2`). Confirm 0 fatal errors, all references resolved, no new overfull hboxes on changed pages. New prose must contain no em dashes and no AI-tell phrasing.

## Monitoring and resume (every hour until done or 09:00)

Three read-only `nvidia-smi` samples before any launch. Every hour, write a measured snapshot to `results/p1_8b_retrain_stronger_20260715/hourly/<timestamp>.json` (stage, GPU snapshot, per-model progress, errors). On an error, diagnose, fix, resume from the last good artifact, and log the fix in a running `fix_log.md`. Never work around an error by fabricating or by silently dropping a model without recording it.

## Acceptance criteria (paste into your closing message)

- `power_diagnostic/rm_set_lock.json` committed before scoring; `power_diagnostic/REPORT.md` with per-reward-model separation results for all ten models and a plain verdict on whether the 8B models are separable.
- Phase B `config.json` with the stronger-training rationale; per-model stability-gate outcomes; the raw-reward power check (do retrained models now move from base?); the validation-split ranked summary with CIs; and, if a signal appeared, the pre-registered fresh sealed-split manifest (unopened).
- The honest 8B power statement landed in `main_v2.tex`, recompiling clean, new prose free of em dashes and AI tells, existing content preserved.
- A one-paragraph bottom line: is the Qwen3-8B model-scale robustness effect measurable under any tested condition, and what is the honest state of the model-scale claim now.
- Hourly snapshots and `fix_log.md`.

## Stop rule

If time runs short, land Phase A and the Phase C honesty edit first, since they are cheap and decisive, and stop Phase B cleanly at the last good artifact. Never re-open the sealed test, never re-select on opened data, never fabricate. An honest "the 8B evaluation is underpowered and stronger training did not change it" is a publishable limitation; a manufactured win is not.
