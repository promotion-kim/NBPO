# Main experiment inputs

The UF-4 rows are a **prospective experiment template**, not completed runs.
The policy training set target is 10,000 prompts. `planned_seeds` is a budget;
`completed_seeds` remains empty until actual runs exist. A missing value is
rendered as `\pending`; it must not be interpreted as zero. Do not put numbers
from the prior SafeRLHF study or unmatched literature protocols in UF-4 rows.

Run from the package directory:

```bash
python3 scripts/render_tables.py
python3 scripts/plot_uf_tradeoffs.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main_v6.tex
```

## CSV files and generated outputs

| Input | Output | Meaning |
|---|---|---|
| `uf_objectives.csv` | `uf_objectives_main.tex` | Independent per-criterion local-judge win rates against the fixed base policy |
| `new_capability_results.csv` | `new_capability_main.tex` | General capability and safety; protocol-specific rates, all in [0,1] |
| `safe_three_seed_results.csv` | `safe_three_seed.tex` | Reported SafeRLHF three-seed aggregates; supplied spread type is not verified |
| `aggregation_policy_pairs.csv` | `aggregation_policy_pairs.tex` | Paired Nash-minus-utilitarian surplus differences and reported intervals |
| `ifeval_per_seed.csv` | Source audit | Reported seed values used to compute the IFEval mean and sample SD |
| `refusal_truncation_audit.csv` | Source audit | Corrected numerator, denominator and conditional quantities |
| `uf_tradeoffs.csv` | `figures/uf_tradeoffs.pdf` and `.png` | Actual four-objective points once observations are supplied |

The earlier CSV files and their generated tables preserve historical results.
New three-seed values do not overwrite an earlier single-seed snapshot.

## Objective table

`if_wr`, `truth_wr`, `honesty_wr`, and `help_wr` are instruction-following,
truthfulness, honesty, and helpfulness **independent evaluation win rates**.
Use the same held-out prompts, judge checkpoint, rubric, base response cache,
position-swap and tie rule for all compared methods. These are not the training
teacher's game values or Nash surpluses. `min_delta` is optional and, when used,
is `min_k (WR_k - 0.5)` under that fixed comparator protocol. It is not a formal
individual-rationality certificate. Fill `eval_prompts` with the actual common held-out prompt count. Record `test_hash`, `judge_id` and
`protocol_id` before populating any result.

Scalarized DPO is one summary row in the compact table. Add explicitly labeled
rows for each alpha/configuration in the CSV used for the trade-off figure;
do not choose a different alpha per metric. The compact table always reports the uniform four-objective weight vector
(0.25, 0.25, 0.25, 0.25), fixed before outcomes are observed.

## Capability table

IFEval uses strict prompt accuracy once verified from raw evaluator output;
GSM8K uses the declared fixed-prompt exact match; MMLU uses a pinned evaluation
protocol. `harmbench_harmful` and `xstest_safe_refusal` are lower-is-better.
Alpaca and both Arena columns are **local proxy** win rates against base,
not official leaderboard scores. Arena hard and creative are kept separate.
The initially supplied three-seed IFEval values have no independently checked
strict-prompt/instruction subtype, which remains an audit item.

The SafeRLHF IFEval means and sample SDs computed from the supplied rounded
seed values are NBPO `0.748 +/- 0.013115` and utilitarian
`0.745 +/- 0.025060`. All other exact method-level aggregate capability cells
remain blank because only family-pooled ranges were supplied. Never assign
a pooled range to every method or infer its mean from its endpoints.

`safe_three_seed_results.csv` preserves supplied aggregate means and spreads.
Its spread type must be confirmed from the raw result generator before
labeling those values SD, SE or confidence intervals. `both_positive_reported`
is stored for audit only because its sampling unit and averaging convention
were not stated. The unverified `both_positive` field is deliberately omitted
from the main table and cannot establish policy-level individual rationality.

## Four-objective figure

One row in `uf_tradeoffs.csv` is one aggregated policy configuration. Fill all
four `*_mean` fields together. Supply its dataset, test hash, protocol ID,
judge checkpoint ID, actual number of prompts and seeds, and explicit seed
IDs. Use a distinct method ID for each scalarization configuration.

Optional `*_ci_low` and `*_ci_high` fields must be paired and accompanied by
`uncertainty_type`; state whether they quantify prompt uncertainty, training
seed variability, or both. Sample SD across three training seeds is not a
confidence interval and must not be inserted into CI endpoint columns.

The figure displays helpfulness against truthfulness and instruction following
against honesty. Filled markers identify nondominated **point estimates among
the evaluated rows using all four criteria**; hollow markers are dominated.
The script does not connect points into a frontier, infer statistical
dominance, or claim global Pareto optimality. Population and model-family
optimality do not follow from a finite comparison set.

Empty input intentionally emits a plainly labeled draft frame with no data
points. Replace it with measured points before submission, or remove it.

## Source limitations

New SafeRLHF numbers originate from the user-provided Claude report dated
2026-09-10 09:13 KST. This package preserves those numbers and checks their
arithmetic; it does not attest to remote training code, run hashes or raw
evaluation artifacts that were not supplied.

The SafeRLHF base row reproduces the latest reported baseline (.752 IFEval,
.867 GSM8K, .281 HarmBench, .500 Alpaca proxy). Its IFEval variant is also
pending verification; the CSV column name specifies the intended future
strict-prompt protocol, not proof that the historical values used it.
