# Game-NBPO: Experiments Required Before ICLR 2027 Submission

## Non-negotiable rule

The existing fixed-anchor NBPO experiments correspond to the \(\beta=\infty\) limit. They cannot be reused as evidence for finite-temperature Game-NBPO. Every main empirical claim must come from runs that actually construct and use the soft criterion adversaries.

## 1. Main matched comparison

Run, on both HelpSteer2 and SafeRLHF:

- Game-NBPO;
- game-utilitarian;
- game-absolute-maxmin;
- game-surplus-maxmin;
- PROSPER / MaxEntBW;
- fixed-anchor NBPO as the \(\beta=\infty\) control;
- MOPO;
- Uniform-INPO;
- the reference policy.

For all game-valued social rules within a seed, hold fixed:

- the certified Phase-0 checkpoint;
- learner response samples;
- reference comparator pool;
- judged pairs and presentation order;
- total judge-query budget;
- total optimizer-update budget;
- temperature vector \(\beta_{1:K}\);
- certification schedule and confidence level;
- evaluator prompts and decoding settings.

External baselines should use their canonical objectives and initialization, but receive the same total query and optimization budget. Hyperparameter-selection budgets must be disclosed and counted.

## 2. Replication

- Minimum: 3 independent training seeds.
- Preferred for a borderline result: 5 independent training seeds.
- Report mean and sample standard deviation across training seeds.
- Do not substitute prompt-level bootstrap intervals for training-seed variation.
- Treat failure to certify a positive-surplus warm start as an observed outcome, not as permission to enter Phase 1 with clipped weights.

## 3. Primary metrics

For every criterion and every seed, report:

- held-out game value \(V_{k,\beta_k}\);
- held-out surplus \(s_k=V_{k,\beta_k}(\pi)-V_{k,\beta_k}(\mu)\);
- minimum surplus;
- Nash welfare \(\sum_k\log s_k\), only when all held-out surpluses are positive;
- individual-rationality rate: fraction of seeds with all surpluses positive.

Also report:

- fixed-reference win rate for every criterion;
- average and worst fixed-reference utility;
- independent overall-quality score;
- response length;
- KL to the deployed reference as a diagnostic;
- criterion-specific adversary entropy and effective sample size;
- accepted/rejected-stage rate;
- certification interval widths;
- total oracle calls and wall-clock cost.

A gain in game value accompanied by a clear loss in ordinary held-out response quality must be described as a trade-off, not as an unconditional success.

## 4. Evaluation separation

Use:

- training preference judges that supply the pairwise labels;
- evaluation judges from different model families;
- a separately generated adversary/comparator pool for evaluation;
- no reuse of the certification batch for final reporting.

The fixed-reference evaluator and the game-value evaluator answer different questions and both are necessary.

## 5. Temperature and comparator-pool ablations

At minimum, evaluate:

- an anchor-like large-\(\beta\) regime;
- one or more intermediate finite temperatures;
- a security-like small-\(\beta\) regime;
- multiple comparator-pool sizes.

For each setting, report:

- policy quality;
- minimum surplus and Nash welfare;
- adversary entropy/effective sample size;
- estimation variance;
- query and compute cost.

The ablation should test the paper's interpolation claim empirically: large \(\beta\) should approach fixed-anchor behavior, while smaller \(\beta\) should react more strongly to exploitable cyclic structure.

## 6. Controlled cyclic test

Include a controlled response-pool experiment in which:

- the full pairwise matrices contain a known cycle;
- a fixed anchor makes the relevant pure responses indistinguishable or nearly indistinguishable;
- the finite-temperature game value distinguishes exploitability;
- Game-NBPO, game-utilitarian, and surplus-maxmin choose measurably different compromise points.

This experiment should validate the central methodological motivation, not merely visualize a synthetic theorem already proved in the paper.

## 7. Scaling/noise intervention

Distinguish two interventions:

1. **Fixed-temperature raw-label scaling:** \(A_k\mapsto\lambda_kA_k\) with \(\beta_k\) unchanged. The effective comparator neighborhood changes, so exact policy invariance is not predicted.
2. **Joint scaling:** \((A_k,\beta_k)\mapsto(\lambda_kA_k,\lambda_k\beta_k)\). The population policy target is predicted to be invariant.

Report:

- target cosine similarity;
- adversary distribution shift;
- normalized criterion weights;
- clean-evaluator utility shift;
- uncertainty across training seeds.

Do not infer policy-level invariance solely from an exact target identity unless the retraining results support it.

## 8. Warm-start and optimizer ablations

Compare:

- full certified Phase 0;
- no certified warm start;
- normalized versus unnormalized inverse-surplus weights;
- monotonic welfare-acceptance gate versus surplus-only gate;
- exact/frozen adversary weights within a stage versus more frequent re-estimation;
- objective-level reference KL as a clearly labeled ablation, not as the main method.

Report both success and failure modes, including rejected stages and certificate failures.

## 9. Main-paper decision rule

Before inserting an empirical claim in the abstract, require all of the following:

- finite-temperature Game-NBPO was actually used;
- the result is reproduced across independent training seeds;
- every game-valued selection rule received a matched budget;
- all criterion coordinates are reported;
- the claimed advantage is not reversed by independent ordinary-quality evaluation;
- the result is stronger than the fixed-anchor control in the setting where cycles matter;
- any lack of improvement on weak-conflict panels is stated directly.

## 10. Tables to fill

The source package contains fill-ready tables for:

- matched finite-temperature main results;
- detailed HelpSteer2 criterion results;
- detailed SafeRLHF criterion results;
- temperature and comparator-pool sensitivity;
- scaling/noise intervention;
- warm-start and optimizer ablations.

Do not remove `TBD` markers until the corresponding run artifacts, seeds, and evaluation files have been archived.
