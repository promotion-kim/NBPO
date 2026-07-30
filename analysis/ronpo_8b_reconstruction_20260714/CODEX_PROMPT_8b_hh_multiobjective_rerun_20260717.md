# CODEX PROMPT — Llama-3.1-8B helpful/harmless re-run: repair the training instrument, add the missing baselines, sweep every arm equally

## 0. Why and what changed

The `p2_8b_hh_multiobjective_20260717` run produced a table that ranked SimPO/IPO/INPO/SPPO above Base and every
RONPO and HT-MNPO arm at or below Base. That is **not a RONPO result. The RONPO arms never trained.** Three
hyperparameters in that run departed from the values established everywhere else in this repository, and all
three selectively disable exactly the arms whose loss is a small-target regression:

| Parameter | Established value | p2 used | Evidence |
|---|---|---|---|
| `learning_rate` | **5.0e-7** | 1.0e-7 | all 28 prior training configs, without exception |
| `ronpo_alpha` | **1.0** | 0.5 | all 3 prior RONPO configs |
| adversary kappa | **0.05** | 0.01 | `ronpo_stage1_local_rm_eval_report.md:110`; `build_os_ronpo_targets.py` default `0.05,0.01` |

Why this is selective rather than a uniform handicap. Every arm in `mnpo_scripts/mnpo_trainer.py` regresses the
policy log-ratio under a squared loss, but the targets differ by four orders of magnitude: INPO/MNPO regress to
`1/(2*eta)` = **66.7**; RONPO to `alpha*zhat` = **±0.5** (±1.0 at the established alpha); HT-MNPO to `eta*gap` =
**±0.0075**; SimPO/IPO use ordinary pairwise losses with well-scaled gradients. Squared-loss gradient is
proportional to the error, so cutting the learning rate 5x freezes the small-target arms while INPO/SimPO/IPO
keep learning. The observed table is a direct readout of that: every small-target arm landed in
0.2414--0.2574 around Base at 0.2599, and every strong-signal arm landed in 0.2831--0.2979. **The p2 table
ranked arms by effective target magnitude, not by robustness.**

Separately, kappa 0.01 made the run's central question unanswerable *by construction*. With
`sigma ∝ sigma0 * exp(-cost/kappa)` and `cost ∈ (0,1)`, kappa 0.01 gives `exp(-100*cost)`: a 0.05 cost gap
between atoms is a mass ratio of `e^-5 ≈ 0.007`, so sigma collapses to a point mass and top-mass, OS, and
full-exp all reduce to the same argmax target. The three RONPO arms duly landed within 0.0085 of each other
with fully overlapping CIs. `PHASE1_run_now.md` already identified this hard-argmax mode as the known-broken
one. At kappa 0.05 the exponent is `exp(-20*cost)` and the aggregation rules genuinely differ, which is why
1.5B stage-2 could separate OS from top-mass.

This re-run repairs the instrument, adds the two baselines the paper's own Table 1 lists but the run omitted,
and gives **every** arm the same hyperparameter search budget.

## 1. Hard rules

- **Never tune, select, or report on "does RONPO win."** Every arm gets the identical sweep budget, the
  identical selection rule, and the identical reporting path, all fixed in section 4 and preregistered before
  any training step. Do not give RONPO a knob, a config, a step count, a seed, or a retry that a baseline does
  not also get. The paper already promises reviewers, in the Table 12 caption, that the worst-objective
  ordering is "not a baseline-tuning artifact"; this run must be able to survive that question.
- **An honest negative is a complete result.** If RONPO does not win the fair sweep, report exactly that.
  A RONPO that only wins when hand-tuned against the metric is not a finding, and reporting one would be worse
  than reporting nothing.
- Restoring the three values in section 0 is a **bug fix against documented repository defaults**, diagnosed
  from a mechanism (sigma collapse; target-scale mismatch), not from the fact that RONPO lost. Do not extend
  this licence to any other parameter. Anything not listed in section 0 keeps its p2 value unless section 4
  sweeps it for every arm.
- Preregister and SHA-256 hash `PREREG.md` + `run_lock.json` before the first training step. Deviations go in
  `fix_log.md`, quoting the frozen text verbatim rather than paraphrasing it.
- Never touch the spent 604-prompt sealed split (`results/p1_sealed_reward_seed42_20260714/`). The 320-prompt
  fresh manifest (SHA-256 `2bf999f00c26bdfeb2bf73a97844dcb13395dc8eef9e65ab03c22f531a14f3d2`) stays sealed
  until the validation table is final; open it once, primary metric only.
- No HuggingFace upload, no paper edits. Regenerate every number from JSON. Report all arms including failures.
- GPUs: the NHN GPUaaS B200 project at `/NHNHOME/WORKSPACE/26msit001_A/mnpo`. Never take another user's GPU.

## 2. Locked setting (unchanged from p2, consumed not re-derived)

Base `meta-llama/Llama-3.1-8B-Instruct`; helpfulness = WildGuard compliance, P(response is not a refusal);
harmlessness = `Qwen/Qwen3Guard-Gen-8B`, log P(safe) - log P(unsafe); the locked 40/20/20/20 mixture
(PKU-SafeRLHF / OR-Bench Hard / XSTest / high-severity BeaverTails). Carry the p2 base deviation declaration
into `PREREG.md` verbatim. Primary metric stays `mean_prompt_worst_norm_score`, the metric the paper already
preregistered. Do not change the base, the objectives, the mixture, or the primary metric.

## 3. Reuse the p2 data if it survived

`results/p2_8b_hh_multiobjective_20260717/train_pool/precompute/` is empty in this workspace; the shared pairs
and log-probabilities may still exist in the NHN project. Check first.

- **If the shared pairs and logps survived**: reuse them byte-for-byte and only rebuild the target columns at
  `--kappas 0.05`. That is CPU-only and takes minutes, and it keeps the re-run exactly comparable to p2.
- **If they did not**: re-decode. Size the pool to **at least 3,000 prompts** this time. The p2 pool was 770
  prompts x 3 pairs ≈ 2,310 rows against 900 steps x effective batch 16 = 14,400 samples, i.e. about 6.6
  epochs, which over-trains a small pool and is a plausible contributor to the uniform drift. Preregister the
  pool size and **report the effective epoch count** in the table's caption either way.

## 4. Arms and the equal-effort sweep

Eleven trained arms plus Base. Two are new; both are in the paper's own Table 1 comparison set, both are
supported by `mnpo_config.py`'s `loss_type` switch, and both were missing from p2:

| Arm | loss_type | Status |
|---|---|---|
| Base | — | untrained reference |
| RONPO (OS) | `ronpo` | `target_os_k0p05` |
| RONPO (top-mass) | `ronpo` | `target_topmass_k0p05` |
| RONPO (full-exp) | `ronpo` | `target_fullexp_k0p05` |
| **MNPO** | `mnpo` | **NEW** — RONPO's direct predecessor (opponent population, heterogeneous); omitting it leaves RONPO's central contrast untested |
| HT-MNPO (harmless) | `ht_mnpo` | `ht_target` |
| HT-MNPO (help.) | `ht_mnpo` | `ht_target_helpfulness` |
| INPO (avg) | `inpo` | |
| SPPO (avg) | `sppo` | |
| **DPO** | `dpo` | **NEW** — the standard direct-alignment baseline; its absence is the first thing a reviewer will notice |
| SimPO | `simpo` | |
| IPO | `ipo` | |

`nhn/run_qwen3_kto_nhn.sh` exists but KTO is not in the trainer's `loss_type` set and not in Table 1; leave it
out of scope and say so.

**The sweep, identical for every arm.** Restore `learning_rate: 5.0e-7`, `ronpo_alpha: 1.0`, kappa `0.05` as the
anchor configuration. Then sweep `learning_rate ∈ {5.0e-7, 1.0e-6}` for **every one of the eleven trained
arms** — two configs each, twenty-two short selection runs at a fixed reduced step count, then one full
900-step run per arm at its selected learning rate. No arm gets a third config. No arm's sweep is skipped
because its anchor "looks fine."

**Selection split.** Select each arm's learning rate on a **tuning split that is disjoint from the 640-prompt
validation panel and from the fresh 320**, by the primary metric, with the identical rule for every arm. The
reporting panel must never be the selection panel. Preregister the tuning split's SHA before sweeping.

If the sweep does not fit the wall clock, cut it **symmetrically**: drop to the anchor learning rate for all
eleven arms and report that no sweep was run. Never sweep a subset of arms.

## 5. Instrument gates (fail-closed, before and during training)

These exist because p2 produced a full table from a broken pipeline and nothing caught it until the end. Each
gate aborts and reports; none of them may be relaxed after seeing a result.

1. **Adversary non-degeneracy, before any training.** Log the normalized entropy of `sigma(k,a)` at the built
   kappa, and verify the three RONPO target columns are actually different: require that
   `target_os_k0p05`, `target_topmass_k0p05`, and `target_fullexp_k0p05` disagree on a substantial fraction of
   rows (report the pairwise identical-fraction and correlation). If sigma is a point mass or the three columns
   are near-identical, **stop**: the RONPO comparison is void by construction and no table may be built. This
   gate alone would have caught p2 before its first training step.
2. **Target-scale audit, before any training.** For every arm, log the effective regression target magnitude
   and the step-0 gradient norm, and put them in the report. "Matched budget" (same rows, same steps, same
   seed) is **not** matched signal — that conflation is exactly what produced the p2 table. If the arms' target
   magnitudes span orders of magnitude, that is a finding to report, not a detail to bury.
3. **Single-oracle sanity, after training.** HT-MNPO (help.) must beat Base on the helpfulness column, and
   HT-MNPO (harmless) must beat Base on the harmlessness column. A method trained on one objective must improve
   that objective. In p2 both failed (help. 0.4700 vs Base 0.5166; harmless 0.4889, below INPO-avg's 0.5187),
   which is what proved the pipeline broken. If either fails again, **the pipeline is broken: report that, and
   do not present a comparison table as if it measured the methods.**

## 6. Evaluation and table

Unchanged from p2: decode every arm on the 640-prompt validation panel, seed 42, matched decode settings; score
with both objectives; rank by `mean_prompt_worst_norm_score`; emit the two-objective table (Help., Harmless,
Avg, Worst with CI, WR_B, wWR_B) plus the LaTeX fragment, with paired 2,000-resample seed-42 bootstrap CIs. Add
two columns the p2 table lacked: **each arm's selected learning rate and its effective target magnitude**, so
the table itself answers the reviewer's tuning question. Report the sweep table for all twenty-two selection
runs. Do not edit the paper.

## 7. Deliverables (into `results/p3_8b_hh_multiobjective_rerun_20260717/`)

- `PREREG.md` + `run_lock.json` (+SHA), carrying the p2 base deviation declaration and the section 0 bug-fix
  rationale with its mechanism.
- `INSTRUMENT.md`: the section 5 gates with their measured values — sigma entropy, the three-column divergence
  check, the per-arm target-scale and step-0 gradient-norm audit, and the single-oracle sanity results.
- `SWEEP.md`: all twenty-two selection runs, the tuning-split SHA, and each arm's selected learning rate.
- `TABLE.md` + LaTeX fragment, all twelve arms including failures, with effective epoch count in the caption.
- `GATE.md`: the preregistered gate, paired bootstrap CI, PASS/FAIL, fail-closed.
- `FRESH.md`: the single fresh-320 confirmation, primary metric only.
- The committed scripts that regenerate every number from JSON.
- `COMPLETION_AUDIT.md`, every score file with row counts and SHA-256, ending `spent_sealed_split_touched=false`.

## 8. One-paragraph brief (paste at top of the codex run)

> Re-run the Llama-3.1-8B helpful/harmless multi-objective training after repairing a broken training
> instrument, and add the two baselines the paper's own Table 1 lists but the previous run omitted. The previous
> run (`p2_8b_hh_multiobjective_20260717`) is void, not a RONPO negative: it used `learning_rate` 1e-7 where all
> 28 prior configs use 5.0e-7, `ronpo_alpha` 0.5 where all 3 prior RONPO configs use 1.0, and adversary kappa
> 0.01 where the established value is 0.05. Because the arms' squared-loss regression targets span four orders
> of magnitude (INPO/MNPO regress to 1/(2*eta)=66.7, RONPO to alpha*zhat=±0.5, HT-MNPO to eta*gap=±0.0075) and
> squared-loss gradient scales with the error, the low learning rate froze the small-target arms while
> SimPO/IPO/INPO kept learning, so the table ranked arms by target magnitude, not robustness; and kappa 0.01
> collapsed the adversary to a point mass, making top-mass, OS, and full-exp the same argmax arm by
> construction. Restore kappa 0.05, ronpo_alpha 1.0, learning_rate 5.0e-7 as the anchor — a bug fix against
> documented defaults, diagnosed from mechanism, not from RONPO losing — and do not extend that licence to any
> other parameter. Keep the locked setting: base Llama-3.1-8B-Instruct, helpfulness = WildGuard compliance,
> harmlessness = Qwen3Guard-Gen-8B log odds, the 40/20/20/20 mixture, primary metric
> mean_prompt_worst_norm_score. Add **DPO** and **MNPO** to the arm list. Reuse the p2 shared pairs and logps if
> they survived in the NHN project and only rebuild targets at kappa 0.05; otherwise re-decode with at least
> 3,000 prompts, and report the effective epoch count either way. Sweep learning_rate over {5e-7, 1e-6} for
> **every one of the eleven trained arms** with an identical budget and an identical selection rule, selecting
> on a tuning split disjoint from the 640-prompt validation panel and the sealed 320; if the wall clock forces a
> cut, drop the sweep for all arms symmetrically and say so. Never give RONPO a knob, config, retry, or step
> that a baseline does not also get: the paper already promises reviewers the worst-objective ordering is "not a
> baseline-tuning artifact". Enforce three fail-closed instrument gates: sigma must not be a point mass and the
> three RONPO target columns must measurably differ before any training; log every arm's effective target
> magnitude and step-0 gradient norm, since matched budget is not matched signal; and after training, HT-MNPO
> (help.) must beat Base on helpfulness and HT-MNPO (harmless) must beat Base on harmlessness, or the pipeline
> is broken and no comparison table may be presented. Preregister and hash everything before the first training
> step, open the fresh manifest once at the end for the primary metric only, never touch the spent 604-prompt
> sealed split, use the NHN B200 project and never another user's GPU, and if RONPO does not win the fair sweep,
> report the honest negative.
