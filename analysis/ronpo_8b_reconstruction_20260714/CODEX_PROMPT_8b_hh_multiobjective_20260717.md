# CODEX PROMPT — Llama-3.1-8B helpful/harmless multi-objective run (RONPO top-mass, RONPO OS, and baselines)

## 0. Why and what changed

The Qwen3-8B scale-up returned an honest negative twice: the robustness effect was not established
(`tab:qwen3-robust-validation` in `ronpo_aaai/main_v3.tex`, where RONPO's worst-objective 0.235 tied INPO),
and the follow-up power diagnostic plus stronger-training check did not rescue it. The diagnosis was a
**measuring-instrument failure, not a method failure**: Qwen3-8B is safety-saturated, so the harmlessness
probe could not resolve, and the general helpfulness reward models (Skywork, Athene, ArmoRM-help) reward
well-written refusals, so helpfulness and safety *agreed* instead of conflicting. A table built on an
instrument with no conflict and no headroom cannot show a worst-case effect whether or not one exists.

The `p1_8b_base_objective_screen_20260716` screen fixed the instrument, inference-only, with the setting
locked **before** any RONPO artifact was trained, decoded, ranked, or scored. It is that locked setting, not a
new search, that this run consumes. This run is the first time RONPO is trained in the repaired 8B setting.

The task here is to train the arms and build the two-objective analogue of Table 4. It is not to look for a
setting in which RONPO wins.

## 1. Hard rules

- **Never select on RONPO outcomes.** The base, both objectives, the prompt mixture, the arm list, the primary
  metric, and the pass gate are all fixed in section 2 and preregistered in section 3 before any training
  step runs. Do not add, drop, swap, or re-tune an arm, a base, an objective, a scorer, a checkpoint rule, or
  a metric because of how RONPO scored. If RONPO loses, that is the deliverable.
- **An honest negative is a complete result.** If RONPO top-mass and RONPO OS do not beat the baselines on the
  preregistered primary metric, report exactly that, keep the table, and state that the 8B scale claim remains
  unestablished and the model-scale claim stays scoped to 1.5B and the synthetic games.
- Preregister and SHA-256 hash `PREREG.md` + `run_lock.json` **before** the first training step. Any deviation
  found later goes in `fix_log.md` with the pre-deviation outputs preserved under `audit/`, and the fix log
  must quote the frozen text it is reconciling against rather than paraphrase it.
- Never touch the spent 604-prompt sealed split (`results/p1_sealed_reward_seed42_20260714/`).
- The 320-prompt fresh-confirmation manifest
  (SHA-256 `2bf999f00c26bdfeb2bf73a97844dcb13395dc8eef9e65ab03c22f531a14f3d2`) stays sealed until every arm is
  trained and the validation table is final. Open it once, for the primary metric only, and never re-open it.
- Matched budget across arms: identical prompt rows, identical reference logps, identical step count,
  identical seed 42, identical optimizer and schedule. Only the loss/target differs.
- No HuggingFace upload. No paper edits. Regenerate every number from JSON. Report all arms, including
  failures. Minimal storage: delete raw generation shards once score JSONL integrity is verified.
- GPUs: the NHN GPUaaS B200 project at `/NHNHOME/WORKSPACE/26msit001_A/mnpo` (see the `nhn/run_qwen3_*.sh`
  launcher pattern: `venv_clean/bin/python`, `accelerate_configs/deepspeed_zero3_port29501.yaml`). On `odin2`,
  GPU 2 was free and GPUs 0-1 were another user's at 23:05 — re-check read-only before use and **never take
  another user's GPU**.

## 2. The locked setting (consumed, not re-derived)

From `results/p1_8b_base_objective_screen_20260716/SELECTION.md`:

| Component | Value |
|---|---|
| Base policy | `meta-llama/Llama-3.1-8B-Instruct` (locked revision in that run's `model_artifact_audit.json`) |
| Helpfulness | WildGuard compliance: continuous P(response is not a refusal) |
| Harmlessness | `Qwen/Qwen3Guard-Gen-8B`: continuous log P(safe) - log P(unsafe) |
| Prompt mixture | 40% PKU-SafeRLHF dual-preference conflicts, 20% OR-Bench Hard should-answer, 20% XSTest (balanced safe/unsafe), 20% high-severity BeaverTails should-refuse |

Screen evidence, all RONPO-agnostic: pooled Spearman between the two objectives **-0.5989** (threshold -0.2);
top-objective set mismatch **0.8891**; helpfulness resolution **0.7695** CI [0.7084, 0.8281]; harmlessness
resolution **5.0938** CI [3.7888, 6.4748]; weaker-objective (compliance) headroom **0.1828** CI [0.1217, 0.2446].
The two objective lineages (AllenAI WildGuard, Qwen Qwen3Guard) are independent.

**Declare this deviation verbatim in `PREREG.md`.** The screen's preregistered rule selects the qualifier with
the most negative rho, which is Qwen2.5-7B + Qwen3Guard8 (rho = -0.6238), not Llama-3.1 (rho = -0.5989). This
run uses Llama-3.1 anyway, for two reasons fixed independently of any RONPO outcome: (i) Llama-3.1 +
Qwen3Guard8 passes every gate under **both** the preregistered `larger`-headroom rule and the post-hoc
`smaller`-headroom variant, so it is the only 8B setting that is robust to that open analysis question; (ii)
Qwen2.5-7B shares the Qwen lineage with the Qwen3Guard harmlessness scorer, and the headroom matrix shows
scorer-lineage affinity (Qwen3Guard scores Qwen2.5 +2.63 over Llama-3.1 while Google's ShieldGemma scores
Llama-3.1 +1.13 over Qwen2.5), which is a live base-scorer confound. State that the rho difference (-0.62 vs
-0.60) is negligible, that the deviation was declared before training, and that it was not made in response to
any RONPO result. Also record the still-open question that `run_selection_analysis.py`'s comparator contradicts
the frozen text in `selection_lock.json` and `PREREG.md`, both of which say `larger`.

## 3. Preregistration (write and hash before the first training step)

- **Arms** exactly as in section 4, with the stretch arms named up front so that adding them later is not a
  new decision.
- **Primary metric**: mean per-prompt worst-objective min-max-normalized score
  (`mean_prompt_worst_norm_score`), normalized per prompt over the eligible model pool, matching
  `analysis/ronpo_8b_reconstruction_20260714/build_ronpo_8b_tables_figure.py`. This is the same primary the
  paper already preregistered, so it is not a new choice.
- **Secondaries**: `mean_objective_norm_score` (Avg), `mean_win_rate_vs_baseline` (WR_B),
  `min_win_rate_vs_baseline` (wWR_B), and the two per-objective normalized columns.
- **Gate**: RONPO OS beats the best non-RONPO arm on the primary with a paired 2,000-resample bootstrap 95% CI
  strictly above zero, seed 42. Fail-closed. Preregister the same test for RONPO top-mass.
- **Normalization scope**: primary and Avg/Worst on the full 640-prompt locked validation panel (consistent
  with how the screen measured conflict and how the paper normalizes). Additionally report, as a secondary
  descriptive only, the sliced breakdown with compliance restricted to should-answer prompts and harmlessness
  restricted to should-refuse prompts.
- **Checkpoint selection rule**: fixed in advance (final step, or a fixed step count), identical for every arm.
  Never select a checkpoint per-arm on the primary metric.

## 4. Arms

Core five, in priority order. Base is untrained.

| Arm | What it is |
|---|---|
| Base | `meta-llama/Llama-3.1-8B-Instruct`, no training; the WR_B/wWR_B reference |
| RONPO (OS) | `target_os_k{K}` from `mnpo_scripts/build_os_ronpo_targets.py` |
| RONPO (top-mass) | `target_topmass_k{K}`, the top-1-truncation arm |
| INPO (avg) | averaging baseline over the two objectives (`nhn/run_qwen3_avg_baseline_nhn.sh`, `nhn/batch_inpo.sh`) |
| HT-MNPO (harmless) | single-oracle on harmlessness (`nhn/run_qwen3_htmnpo_nhn.sh` with the objective list set to the two new scorers) |

Stretch, added only in this order and only if the section 7 checkpoints are met early:
`HT-MNPO (help.)` first (it is what makes the worst-objective claim two-sided: each single-oracle arm should be
weak on the objective it ignores), then `SPPO (avg)`, `SimPO`, `IPO`, then `RONPO (full-exp)`.

`mnpo_scripts/build_os_ronpo_targets.py` recomputes the top-mass, OS, and full-exp targets analytically from an
**existing** precomputed dataset, reusing its (chosen, rejected) pairs and logps with no re-decode and no GPU.
Exploit this: pay the decode and scoring cost once, then the RONPO arms differ only in a scalar target column,
which is exactly the matched budget the paper claims. Do not build them from separate decodes.

## 5. Data

Build three prompt-disjoint splits from the locked mixture sources, deduplicated against each other and against
both existing manifests:
- **train**: fresh prompts, disjoint from the 640 validation and 320 fresh manifests. Size it to what the
  section 7 budget allows (target roughly 5k prompts) and record the count and SHA before training.
- **validation**: the existing 640-prompt locked panel, SHA-256
  `1d21a88603095af464fa64a057736c4717ec4ac69a377ffe1720378f19319ce9`. Used for the table and any selection.
- **fresh confirmation**: the sealed 320-prompt manifest, opened once at the end for the primary metric only.

Decode on-policy from the base with its own chat template, seed 42. Score every response with both objectives.
Record source revisions, counts, dedup procedure, and hashes in `dataset_manifest/`.

## 6. Evaluation and table

Decode every arm on the 640-prompt validation panel, seed 42, matched decode settings. Score with WildGuard
compliance and Qwen3Guard8. Then emit the two-objective analogue of `tab:qwen3-robust-validation` (drop the
Concise column):

```
Method              | Help.  Harmless | Avg    Worst   WR_B    wWR_B
```

Rank rows by the primary (`mean_prompt_worst_norm_score`), bold best and underline second, exactly as
`build_ronpo_8b_tables_figure.py` already does. Report the paired bootstrap CI for the preregistered gate.
Ship the LaTeX fragment but **do not edit the paper**.

## 7. Time budget (hard) and fail-safes

Deadline **09:00 KST 2026-07-17**; it is roughly 23:10 now, so about 9.8 hours. Checkpoints, each fail-closed:

| By | State |
|---|---|
| 00:10 | `PREREG.md` + `run_lock.json` hashed; splits built and hashed; scorer smoke test passed |
| 02:30 | train decode + both scorers done; precomputed dataset + logps + all three RONPO target columns built |
| 06:30 | all five core arms trained (run two arms concurrently if the B200 memory allows) |
| 08:00 | validation decode + scoring for all arms done |
| 08:30 | validation table + gate CI regenerated from JSON |
| 08:45 | fresh-320 opened once, primary metric only; `COMPLETION_AUDIT.md` written |

If a checkpoint slips, **cut scope, never quality**: drop stretch arms first, then reduce train prompts, then
report the table with the arms that finished and mark the missing ones explicitly. If an arm crashes, report it
as crashed. Never impute, extrapolate, or fabricate a number, and never quietly shrink the panel. If nothing
trains in time, deliver the preregistration, the data manifest, and an honest status note by 09:00.

## 8. Deliverables (into `results/p2_8b_hh_multiobjective_20260717/`)

- `PREREG.md` + `run_lock.json` (+SHA), including the section 2 base deviation declaration.
- `dataset_manifest/` with all three split hashes and the dedup record.
- `TABLE.md` + the LaTeX fragment for the two-objective table, with the full arm list including failures.
- `GATE.md`: the preregistered gate, the paired bootstrap CI, and PASS/FAIL, fail-closed.
- `FRESH.md`: the single fresh-320 confirmation of the primary metric.
- The committed scripts that regenerate every number from the score JSONL.
- `COMPLETION_AUDIT.md`, listing every score file with row counts and SHA-256, and ending
  `spent_sealed_split_touched=false`.

## 9. One-paragraph brief (paste at top of the codex run)

> Train the helpful/harmless multi-objective arms in the repaired 8B setting that the
> `p1_8b_base_objective_screen_20260716` screen already locked, and build the two-objective analogue of the
> paper's Table 4. The setting is fixed and must not be re-searched: base `meta-llama/Llama-3.1-8B-Instruct`,
> helpfulness = WildGuard compliance (P not a refusal), harmlessness = `Qwen/Qwen3Guard-Gen-8B` safe-vs-unsafe
> log odds, prompt mixture 40% PKU-SafeRLHF / 20% OR-Bench Hard / 20% XSTest / 20% high-severity BeaverTails.
> That screen was inference-only and RONPO-agnostic (conflict rho -0.5989, two-sided resolution, compliance
> headroom CI [0.1217, 0.2446]); declare in the prereg that using Llama-3.1 deviates from the screen's
> most-negative-rho rule (which picked Qwen2.5-7B at -0.6238), that the reasons are robustness to the open
> `larger`-vs-`smaller` headroom question and avoiding the Qwen-base/Qwen-scorer lineage confound, and that the
> deviation was declared before training and not in response to any RONPO result. Core arms: Base, RONPO (OS),
> RONPO (top-mass), INPO (avg), HT-MNPO (harmless); stretch in order HT-MNPO (help.), SPPO (avg), SimPO, IPO,
> RONPO (full-exp). Use `mnpo_scripts/build_os_ronpo_targets.py` so the RONPO arms share one decode, one set of
> logps, and one row set, differing only in the scalar target, which is the matched budget the paper claims.
> Primary metric is the already-preregistered mean per-prompt worst-objective normalized score; gate is RONPO OS
> over the best non-RONPO arm with a paired 2,000-resample seed-42 bootstrap 95% CI above zero, fail-closed.
> Preregister and hash everything before the first training step. Never re-select the base, objectives, arms,
> metric, or checkpoint rule on RONPO's results; if RONPO loses, report the honest negative and keep the scale
> claim scoped to 1.5B. Train on fresh prompts disjoint from the 640-prompt validation panel and the sealed
> 320-prompt fresh manifest; open the fresh manifest once at the very end for the primary metric only; never
> touch the spent 604-prompt sealed split. Use the NHN GPUaaS B200 project and never another user's GPU.
> Deadline 09:00 KST: cut scope, never quality, and never fabricate a number.
