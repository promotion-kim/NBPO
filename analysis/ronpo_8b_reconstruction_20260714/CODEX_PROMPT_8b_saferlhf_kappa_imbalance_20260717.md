# CODEX PROMPT — SafeRLHF helpful/harmless: does any adversary temperature make RONPO robust under objective imbalance?

## 0. What this run is, and what it is not

This run asks one question: **does RONPO's adversary buy worst-objective robustness when the two objectives are
imbalanced, and at what adversary temperature kappa?** The hypothesis under test is mechanistic and directional:
when most training prompts carry no harmlessness signal, an averaged oracle should spend its capacity on
helpfulness and let harmlessness rot, while RONPO's adversary should keep reweighting toward the neglected
objective. **If that is real, RONPO's worst-objective advantage must grow with imbalance and be approximately
zero at balance.**

That last clause is the point. **This run preregisters a null at the balanced condition.** A design that can
only produce "RONPO wins" is not a test of the hypothesis; it is a test of the sweep budget. The prior two
positive RONPO results in this repository (8B calibration signal, 1.5B stage-2 top-mass) both failed to
replicate out-of-sample and were diagnosed as winner's curse. This run is built so that the same failure cannot
be reported as a success.

**This is not a "sweep kappa until RONPO wins" run.** Section 4 fixes exactly one confirmatory RONPO
configuration by a reward-blind criterion before any training. The kappa curve is reported in full as mechanism
evidence and is explicitly non-confirmatory.

## 1. Hard rules

- **An honest negative is a complete result.** If RONPO's advantage does not grow with imbalance, report exactly
  that. A RONPO that only wins at a kappa chosen after seeing the metric is not a finding.
- **One confirmatory RONPO arm, fixed before training, by a reward-blind rule** (section 4). Every other cell in
  the kappa grid is a diagnostic curve, labeled non-confirmatory in every table it appears in.
- **Report `best-kappa RONPO` explicitly as a winner's-curse-inflated upper bound**, never as the headline
  number, and never in the same column as the confirmatory arm. State in the caption that it was selected on the
  reporting panel and is therefore biased upward.
- Preregister and SHA-256 hash `PREREG.md` + `run_lock.json` before the first training step. Deviations go in
  `fix_log.md`, quoting the frozen text verbatim.
- Never touch the spent 604-prompt sealed split (`results/p1_sealed_reward_seed42_20260714/`). Do not reuse the
  p1/p2 validation-256 or fresh-128 manifests as *training* prompts; they stay held out.
- No HuggingFace upload, no paper edits. Regenerate every number from JSON. Report all arms including failures.
- GPUs: the NHN GPUaaS B200 project (8x B200) at `/NHNHOME/WORKSPACE/26msit001_A/mnpo`. Never take another
  user's GPU.

## 2. Three facts verified before this prompt was written. Do not re-derive them; do not violate them.

**(a) The top-mass target is exactly invariant to kappa. A top-mass kappa sweep is a no-op.**
In `mnpo_scripts/build_os_ronpo_targets.py:62`, top-mass takes `zhat` at `argmax(sigma)` where
`sigma ∝ exp(-cost/kappa)`. That function is monotone decreasing in `cost` for every kappa > 0, so
`argmax sigma = argmin cost` at every temperature. Verified numerically over 2,000 rows (K=2, A=4, scale=8)
across kappa ∈ [0.01, 0.5]: **the top-mass target varies in 0/2000 rows; OS and full-exp vary in 2000/2000.**
Training top-mass at N kappas would train N identical models and report them as a temperature curve.
**Top-mass enters the grid exactly once, as the kappa-invariant hard-argmax anchor.**

**(b) kappa has two degenerate limits, and both collapse RONPO onto an existing arm.**
Measured on the same synthetic (normalized entropy of sigma over K*A=8 atoms; correlation of the full-exp target
with the two endpoints):

| kappa | norm. entropy of sigma | corr(full-exp, top-mass) | corr(full-exp, uniform) |
|---|---|---|---|
| 0.001 | 0.009 | **0.998** | 0.677 |
| 0.01 (p2's value) | 0.084 | **0.982** | 0.708 |
| 0.02 | 0.156 | 0.965 | 0.743 |
| 0.05 (repo default) | 0.331 | **0.913** | 0.835 |
| 0.1 | 0.543 | 0.834 | 0.920 |
| 0.2 | 0.783 | 0.754 | 0.970 |
| 0.5 | 0.953 | 0.704 | 0.994 |
| 100 | 1.000 | 0.673 | **1.000** |

As kappa → 0 sigma becomes a point mass and OS = full-exp = top-mass (this is why p2's three RONPO arms landed
within 0.0085 of each other). As kappa → ∞ sigma becomes uniform and the target becomes `mean_{k,a} zhat(k,a)`,
i.e. **RONPO with the adversary switched off**. RONPO can only be distinct in the interior.

**(c) This repository has never once explored the soft-adversary regime.** Every RONPO run to date used
kappa ≤ 0.05, i.e. normalized entropy ≤ 0.33 and ≥ 0.91 correlation with hard argmax. The entire kappa > 0.05
half of the mechanism is unmeasured. That, not a finer grid near 0.01, is where the new information is.

**Consequence for the grid:** the kappa curve interpolates between two arms this run trains exactly:
top-mass (the exact kappa→0 limit) and RONPO-uniform (the exact kappa→∞ limit, the no-adversary control).
**RONPO's claim is that soft-robust aggregation beats both endpoints — an interior peak.** That is a falsifiable
*shape*, and it is far stronger than any single win. If the curve is monotone toward either endpoint, RONPO's
adversary is not buying anything and the honest conclusion is that the estimator reduces to a known method.

**Add an exact `target_uniform` column** to `build_os_ronpo_targets.py` (= `zhat.mean()`, no kappa). This is the
no-adversary ablation: same loss, same rows, same target scale, same budget, adversary removed. It is a far
cleaner control than INPO-avg, which differs in loss form, target scale, and pair orientation simultaneously.

## 3. The contradiction this run adjudicates

Two existing results disagree about kappa=0.05, and this run is the tiebreak. Preregister both predictions.

- **`toy/os_ronpo_kappa_sweep.py`, re-run 2026-07-17 (output in `TOY.md`):** on the hard worst-objective floor,
  **top-mass BEATS the OS/full-exp Nash floor at kappa=0.05** (0.3014 vs 0.2881, i.e. −0.0133 for OS). OS only
  overtakes top-mass at kappa ≤ 0.005 (max margin +0.0052 at kappa=0.002), because the regularized Nash
  approaches the unregularized robust LP value V*=0.3289 only as the entropic regularization vanishes.
- **1.5B stage-2, measured 2026-07-16:** at kappa=0.05, **OS BEATS top-mass** on worst-objective
  (0.5004 vs 0.4840 vs base 0.4144; OS-vs-base CIs disjoint).

These cannot both be the whole story. Note the trap: the toy's OS-favourable region (kappa ≤ 0.005) is exactly
where fact (b) says the *static* target builder collapses OS onto top-mass (entropy 0.05, corr 0.998). So **if
the toy's mechanism is the true one, the current implementation cannot realize it**, and that is itself a
publishable finding about the estimator rather than a tuning failure.

A likely reconciliation to state in `PREREG.md` and test, not assume: **the trainer's adversary is not the toy's
adversary.** The toy runs OMD, so sigma adapts to the policy every step. `build_os_ronpo_targets.py` computes
sigma **once, analytically, with cost evaluated under pi = uniform (the frozen base pool)** — a static,
policy-blind adversary that never responds to training. The toy's saddle-point convergence story does not
transfer to a fixed target column. Do not paper over this. Report it in `INSTRUMENT.md`.

## 4. The confirmatory arm, fixed by a reward-blind rule, before training

**The kappa grid is chosen by sigma entropy, not by outcome.** On the real precomputed data (not the synthetic
table above), compute the empirical map from kappa to mean normalized entropy of sigma. Then pick the five
kappas whose entropies land nearest **{0.05, 0.15, 0.35, 0.55, 0.85}**. This spans degenerate-argmax through
soft, uses no reward, no policy, and no training, and is computable in minutes on CPU. Freeze and hash the
resulting kappa list before any training step.

**The single confirmatory RONPO arm is OS at the entropy≈0.55 kappa.** Fixed now, before any model exists,
because it is the midpoint of the non-degenerate range — a geometric criterion, not an outcome. Every other
kappa is diagnostic.

**Why this is fair even though RONPO gets a 5-point curve and each baseline gets one config.** RONPO's
*reported confirmatory* configuration count is one, identical to every baseline. The curve is an instrument
reading, not a selection. Any table that shows a kappa other than the confirmatory one must label the row
`diagnostic (non-confirmatory)`. The paper already promises reviewers in the Table 12 caption that the
worst-objective ordering is "not a baseline-tuning artifact"; this run must survive that question.

## 5. The imbalance axis — the independent variable

**rho = fraction of training prompts that are PKU-benign**, i.e. prompts where PKU's human annotators labeled
*both* Alpaca3-8B responses safe (`is_response_0_safe == True and is_response_1_safe == True`). The complement
("safety-active") is any prompt where at least one response was labeled unsafe.

This lever is chosen because it is **external, policy-blind, and reward-model-blind**: it is a human annotation
from PKU about the prompt's risk, carried by fields this repository has never read. It cannot be contaminated by
our Llama decode or our Beaver scorers, so it cannot manufacture the effect it is used to test.

**Stage 1 (run first): rho ∈ {0.5, 0.9}.** Two levels is all the interaction needs. Preregistered predictions:

1. **rho=0.9 (imbalanced):** the confirmatory RONPO arm beats the best averaged-oracle baseline on
   `mean_prompt_worst_norm_score`, lower 95% bound of the paired difference > 0.
2. **rho=0.5 (balanced): RONPO does NOT beat it.** Predicted null. If RONPO wins here too, the mechanism story
   is *wrong even if the headline is positive* — it would mean the advantage is not about imbalance — and the
   report must say so instead of claiming a win.
3. **Component-level (the diagnostic that distinguishes mechanism from luck):** as rho rises, the *baselines'*
   harmlessness column must fall while RONPO's holds. The aggregate can move for many reasons; this specific
   decomposition is what the mechanism predicts and nothing else does.

**Stage 2, conditional on Stage 1 showing the interaction: add rho=0.75** to test monotonicity. If Stage 1 is
null, stop and report the negative; do not add levels hunting for a cell that works.

Row counts must be **identical across rho** — only composition changes, never budget. Cap N at the largest value
constructible at every rho from the available pool, and preregister it.

## 6. Data — and the bottleneck that has silently starved every prior run

**PKU-SafeRLHF is already cached locally; do not re-download the test split.**
`results/p1_8b_base_objective_screen_20260716/source_cache/pku_saferlhf/data/Alpaca3-8B/test.jsonl`
(sha256 `7f7ee8812fbeb52e1568a2b91d1d90bf6d0064b88ba8362198a7234d30007781`, 2,327 rows, revision
`9421ffafec3fa40a1f1a7d567b4d525079477ecb`). Verified contents:

| slice | count |
|---|---|
| total rows / unique prompts | 2,327 / 2,301 |
| dual-preference conflict (`better != safer`) | 696 (693 unique) |
| **both responses safe (benign)** | **949** |
| exactly one unsafe | 348 |
| both unsafe | 1,030 |

**The bottleneck: every prior run loaded only `test.jsonl` of a single subset.** p2's `PREREG.md:20` justifies
its 770-prompt pool by saying only "309 unused PKU dual-preference conflicts" exist. That is not a property of
PKU-SafeRLHF; it is a property of having read 2,327 of its ~83k rows. The p3 prompt separately flags that the
770-prompt pool meant ~6.6 epochs and probably contributed to uniform drift. **Both problems have the same
one-line fix.**

1. **Download `data/Alpaca3-8B/train.jsonl` at the same pinned revision** and use it as the training pool.
   Report its row count and its benign/safety-active split in `DATA.md`. Expect roughly an order of magnitude
   more prompts, which is what makes a 3,000+ pool and a matched-N imbalance sweep possible at all.
2. **Keep the test split for the validation and fresh panels.** Dedup by normalized prompt across train/test and
   against the p1/p2 held-out manifests; report the collision count.
3. **Drop the conflict-only filter.** Prior builders keep only `better != safer` rows. That selects for maximal
   tension and makes an imbalance sweep impossible by construction, since benign prompts are exactly what rho=0.9
   needs. The pool is now stratified by the benign/safety-active label instead. Say so explicitly in `PREREG.md`
   as a deliberate, preregistered departure from the p1/p2 slice definition, with this reason.
4. `analysis/p2_8b_hh_multiobjective_20260717/build_shared_pairs.py` hardcodes 4 responses/prompt (line 72) and
   770 rows (line 59). Parameterize both; do not work around them.

## 7. Stage 0 — pre-flight gates. CPU and one GPU. Nothing on the B200 queue until all three pass.

Every prior 8B run in this repository died of **evaluator power**, not of method: ArmoRM heads saturated, all
ten policies scored within 0.005, and nobody found out until after the GPU spend. These gates cost under an hour
and are the entire reason to prefer SafeRLHF + Beaver.

1. **Scorer calibration against human labels — the check every prior run lacked.**
   PKU ships `response_0` and `response_1` *and* the two per-objective human verdicts, and this repository has
   never read any of them. Score both human responses with both Beaver models and check agreement:
   - `PKU-Alignment/beaver-7b-v1.0-cost@c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28` (harmlessness, score `-cost`)
     must predict `safer_response_id` **on the 696 conflict rows** at **≥ 65%**.
   - The helpfulness scorer must predict `better_response_id` on the same rows at **≥ 60%**.
   - Conflict rows are the discriminating set on purpose: a scorer that merely tracks overall quality scores
     ~50% there, and that is precisely the failure mode that produced four consecutive 8B nulls.
   - Also report agreement on non-conflict rows and the score spread on the decoded pool.
   **If either scorer cannot beat these thresholds, the objectives are noise. Stop and report. Do not train.**
2. **Helpfulness scorer resolution.** `on_policy_data_gen/rm_beaver_cost.py` scores a Safe-RLHF *cost* model;
   the matching *reward* model is a separate checkpoint. Confirm the exact helpfulness scorer and its revision,
   pin it in `run_lock.json`, and verify it is a distinct head from the cost model — not the same backbone
   scored twice, which would make the two "objectives" one objective and guarantee a null.
3. **Synthetic pre-flight of the directional prediction.** Before any B200 hour, reproduce the imbalance
   prediction in the `toy/` harness: sweep the fraction of objective-inactive rows and confirm that RONPO's
   worst-objective margin over the uniform-adversary control grows with it. **If the interaction does not appear
   in a synthetic world where imbalance is exactly known and there is no evaluator noise, it will not appear at
   8B.** A negative here is a cheap, complete, reportable result and should stop the run.

## 8. The grid and the B200 plan

**Do the expensive work once.** Decode the union pool once (Llama-3.1-8B-Instruct, 4 responses/prompt, seeds
42/43/44/45, temp 0.7, top-p 0.9, 1,024 new tokens, bf16). Score once with both Beaver heads. Build pairs and
log-probabilities once. **Then:**
- **All kappas are CPU-only and cost minutes:** one `build_os_ronpo_targets.py --kappas <the five> ` pass adds
  every OS/full-exp target column plus `target_uniform` to the same dataset. The kappa sweep is nearly free.
- **Imbalance is a row filter, not a re-decode:** every rho is a matched-N subsample of the same precomputed
  rows. No extra decode, no extra precompute, and the arms stay byte-comparable across rho.

**Arms, per rho (20 trained + Base):**

| Arm | loss_type | target column | role |
|---|---|---|---|
| Base | — | — | untrained reference (decode once, shared across rho) |
| RONPO OS x 5 kappas | `ronpo` | `target_os_k*` | **1 confirmatory** (entropy≈0.55) + 4 diagnostic |
| RONPO full-exp x 5 kappas | `ronpo` | `target_fullexp_k*` | diagnostic curve |
| RONPO top-mass | `ronpo` | `target_topmass_*` | exact kappa→0 endpoint. **One config only — fact (a)** |
| RONPO uniform | `ronpo` | `target_uniform` | exact kappa→∞ endpoint = **no-adversary control** |
| INPO-avg, SPPO-avg, MNPO | `inpo`/`sppo`/`mnpo` | — | averaged-oracle family |
| HT-MNPO (help.), HT-MNPO (harmless) | `ht_mnpo` | `ht_target*` | single-oracle references |
| DPO, SimPO, IPO | `dpo`/`simpo`/`ipo` | — | standard direct-alignment baselines |

**Anchor hyperparameters** (the p3 bug-fix values; do not re-break them): `learning_rate` **5.0e-7**,
`ronpo_alpha` **1.0**, `ronpo_tau` 0.05, eta 0.0075, seed 42, 900 steps, effective batch 16, reference-anchor
0.05, preference-SFT 0.005, bf16. Report the effective epoch count in every table caption.

**B200 fan-out: 8 GPUs, one arm per GPU.** Use the `single_gpu.yaml` pattern from
`nhn/run_b200_conflict_matched_train.sh:41-56` / `analysis/p2_8b_hh_multiobjective_20260717/train_core_arms.py`
(8B fits one B200 at `per_device_train_batch_size=1`, `gradient_accumulation_steps=16`), driven from a
`name → gpu` TSV, backgrounded, `wait`-ed. Stage 1 is 40 runs = 5 waves of 8; Stage 2 adds 20 = 2.5 waves.
Decode with 8 independent single-GPU vLLM workers (`--num_gpu 1`, `MAX_PARALLEL_DECODE=8`), not TP=8.
**Never leave a B200 idle waiting on a barrier**: queue the next wave's arms as GPUs free, since all arms are
independent given the shared precompute. Log per-wave GPU-hours and idle time in `COMPLETION_AUDIT.md`.

## 9. Instrument gates (fail-closed, none relaxable after seeing a result)

1. **Adversary non-degeneracy, before training.** Log normalized sigma entropy at every built kappa and the
   pairwise identical-fraction and correlation among `target_topmass`, `target_os_k*`, `target_fullexp_k*`,
   `target_uniform`. **Assert the measured curve reproduces facts (a) and (b):** top-mass identical across all
   kappas (fraction differing must be 0.0), and full-exp's correlation with top-mass falling monotonically while
   its correlation with `target_uniform` rises. If the endpoints do not bracket the curve, the target builder is
   broken and no table may be built.
2. **Target-scale audit, before training.** Per arm, log the effective regression target magnitude and step-0
   gradient norm. Matched budget is not matched signal — conflating the two is exactly what produced the void p2
   table (targets spanning four orders of magnitude: INPO 66.7, RONPO ±1.0, HT-MNPO ±0.0075).
3. **Single-oracle sanity, after training.** HT-MNPO (help.) must beat Base on helpfulness and HT-MNPO
   (harmless) must beat Base on harmlessness, **at every rho**. A method trained on one objective must improve
   that objective. In p2 both failed, which is what proved the pipeline broken. If either fails, the pipeline is
   broken: report that and **do not present a comparison table as if it measured the methods**.
4. **Reward-blind stability gate**, per `stability_gate_corrected.py`: exact response count, zero non-empty think
   spans, mean-word ratio to base in [0.33, 2.0], max consecutive identical-word run ≤ 20. Gate on the **full**
   validation panel, never a 128-prompt subset — under-sampling collapse triggers is a known, diagnosed failure
   of the 2026-07-16 stage-2 run. Failures are reported and excluded fail-closed; thresholds never move.

## 10. Evaluation and the tables

Decode every arm on the validation panel, seed 42, matched decode settings. Score with both Beaver heads.
Primary metric `mean_prompt_worst_norm_score` (mean over prompts of the min of the two per-prompt min-max
normalized objectives). Paired 2,000-resample seed-42 bootstrap CIs.

Emit, all regenerated from JSON:
- **`TABLE_main.md`** — per rho: Help., Harmless, Avg, Worst with CI, WR_B, wWR_B, for all arms including
  failures. The **confirmatory** RONPO row is marked; every other kappa row is marked
  `diagnostic (non-confirmatory)`. Include each arm's effective target magnitude, and the epoch count in the
  caption.
- **`CURVE.md`** — worst-objective vs sigma entropy at each rho, with the two endpoint arms (top-mass,
  RONPO-uniform) drawn as horizontal references. **This plot is the run's central scientific object.** State
  plainly which of the three shapes it shows: interior peak (RONPO's claim holds), monotone toward top-mass
  (the estimator reduces to hard argmax), or monotone toward uniform (the adversary buys nothing).
- **`INTERACTION.md`** — the preregistered test: RONPO-minus-best-averaged-baseline worst-objective difference
  at each rho, with CIs, plus the per-objective decomposition from section 5.3. **Report the rho=0.5 null as a
  result**, not as a disappointment.
- **`TOY.md`** — the section 3 contradiction: the toy's kappa=0.05 prediction, the 1.5B measurement, and which
  one the 8B curve supports. If neither, say so.

Do not edit the paper. Do not open any fresh confirmation split until the validation tables and the section 5
predictions are final; then open once, primary metric only.

## 11. Deliverables (into `results/p4_8b_saferlhf_kappa_imbalance_20260717/`)

- `PREREG.md` + `run_lock.json` (+SHA): the frozen kappa list with its entropy criterion, the confirmatory arm,
  the rho levels and matched N, the three section-5 predictions **including the rho=0.5 null**, the section-6
  slice departure with its reason, and the scorer pins.
- `PREFLIGHT.md`: the three section-7 gates with measured values — Beaver-vs-human agreement on conflict rows,
  scorer-distinctness, and the synthetic interaction check.
- `DATA.md`: train-split row counts, benign/safety-active split, dedup collisions, matched N per rho, SHAs.
- `INSTRUMENT.md`: section-9 gates with measured values, and the section-3 static-vs-adaptive adversary finding.
- `TABLE_main.md`, `CURVE.md`, `INTERACTION.md`, `TOY.md` + LaTeX fragments.
- `GATE.md`: the preregistered gate, paired bootstrap CI, PASS/FAIL, fail-closed.
- The committed scripts that regenerate every number from JSON, including the `target_uniform` patch to
  `build_os_ronpo_targets.py` and the parameterization of `build_shared_pairs.py`.
- `COMPLETION_AUDIT.md`: every score file with row counts and SHA-256, per-wave GPU-hours and idle time, ending
  `spent_sealed_split_touched=false`.

## 12. One-paragraph brief (paste at the top of the codex run)

> Test whether RONPO's adversary buys worst-objective robustness when helpfulness and harmlessness are
> imbalanced, on PKU-SafeRLHF with Llama-3.1-8B-Instruct and Beaver reward/cost objectives, on the NHN 8x B200
> project. Three facts are already verified and must not be re-derived or violated. **(a) The top-mass target is
> exactly kappa-invariant** — `argmax exp(-cost/kappa) = argmin cost` for every kappa, confirmed 0/2000 rows
> varying across kappa ∈ [0.01,0.5] while OS and full-exp vary 2000/2000 — so top-mass enters the grid **once**,
> and any top-mass kappa sweep would train identical models. **(b) kappa has two degenerate limits**: as kappa→0
> sigma is a point mass and OS = full-exp = top-mass (why p2's three RONPO arms landed within 0.0085), and as
> kappa→∞ sigma is uniform and the target becomes RONPO-with-the-adversary-off; add an exact `target_uniform`
> column as that no-adversary control. **(c) This repo has never tested kappa > 0.05** (entropy ≤ 0.33, ≥0.91
> correlated with hard argmax), so the soft half of the mechanism is unmeasured. RONPO's real claim is therefore
> an **interior peak** on a curve interpolating top-mass and RONPO-uniform — a falsifiable shape, not a win.
> Choose the five-kappa grid by **sigma entropy targets {0.05,0.15,0.35,0.55,0.85} measured on the real data**
> (reward-blind, CPU, minutes), and fix **one confirmatory arm — OS at entropy≈0.55 — before any training**;
> every other kappa is a labeled diagnostic, and any best-kappa number is reported only as a winner's-curse upper
> bound. The independent variable is **rho, the fraction of PKU-benign prompts** (both human responses labeled
> safe) — an external, policy-blind, reward-blind human label this repo has never read. Preregister three
> predictions: RONPO wins at rho=0.9; **RONPO does NOT win at rho=0.5 (a preregistered null)**; and as rho rises,
> baselines' harmlessness falls while RONPO's holds. Run rho ∈ {0.5,0.9} first; add 0.75 only if the interaction
> appears. **Unlock the data bottleneck**: every prior run read only `data/Alpaca3-8B/test.jsonl` (2,327 rows,
> 949 benign), which is why p2 claimed only "309 available conflicts" and trained 6.6 epochs on 770 prompts —
> download `train.jsonl` at the same pinned revision, keep test for validation, and **drop the conflict-only
> filter**, since benign prompts are exactly what rho=0.9 requires. **Before any B200 hour, pass three
> pre-flight gates**: Beaver-cost must predict PKU's human `safer_response_id` on the 696 conflict rows at ≥65%
> and the helpfulness head must predict `better_response_id` at ≥60% (every prior 8B run died of evaluator
> saturation, and this check — using response_0/response_1, fields this repo has never read — costs an hour);
> the two heads must be genuinely distinct; and the imbalance interaction must first reproduce in the `toy/`
> harness, since if it fails where imbalance is exactly known and noise is zero it will not appear at 8B. Note
> and report the tension the run adjudicates: the toy says top-mass **beats** OS at kappa=0.05 (0.3014 vs
> 0.2881) and OS only wins at kappa ≤ 0.005 — exactly where the static builder collapses OS onto top-mass —
> while 1.5B stage-2 measured OS **beating** top-mass at kappa=0.05 (0.5004 vs 0.4840); a likely cause is that
> the toy's adversary adapts via OMD while `build_os_ronpo_targets.py` freezes sigma once under pi=uniform, so
> report that rather than paper over it. Decode, score, and precompute **once**; all kappas are CPU column-adds
> and every rho is a matched-N row filter, so fan 20 arms/rho across the 8 B200s one-arm-per-GPU and never idle
> a GPU on a barrier. Enforce fail-closed gates: the measured entropy curve must reproduce facts (a) and (b);
> log every arm's target magnitude and step-0 gradient norm, since matched budget is not matched signal; HT-MNPO
> single-oracle sanity must hold at every rho or the pipeline is broken and no table may be presented; and gate
> stability on the **full** panel, never a 128-prompt subset. Preregister and hash everything first, never touch
> the spent 604-prompt sealed split, and **if RONPO's advantage does not grow with imbalance, report the honest
> negative** — including the case where RONPO wins at rho=0.5 too, which would refute the mechanism even if the
> headline looks positive.
</content>
</invoke>
