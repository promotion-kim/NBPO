# NBPO v6: general-instruction 8B main experiments

This is a prospective experiment manuscript and a reproducible table/figure template, updated with the reported SafeRLHF three-seed comparison. The latest supplied main_v6(1).tex is the edit base. The core mathematics, algorithm and proof structure are retained. New UF-4 measurements are blank and the trade-off figure is explicitly a draft template.

The compiled manuscript has 26 pages, with the conclusion on page 9. The style is unchanged. The bibliography adds one ModernBERT reference. The main experiment sequence is four-objective policy outcomes, empirical trade-offs, general capability/safety, then the completed SafeRLHF comparison. Historical controlled/realization/trajectory/pool diagnostics remain in the appendix.

## Build and populate

```bash
python3 scripts/render_tables.py
python3 scripts/plot_uf_tradeoffs.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main_v6.tex
```

Use TeX Live and Python NumPy/Matplotlib. Included algorithm/algorithmic files are unmodified CTAN copies with sources under vendor/algorithms.

Read data/MAIN_EXPERIMENTS_SCHEMA.md before filling the new main inputs. Edit CSVs rather than generated TeX. The compact DPO row always uses uniform weights. The figure evaluates descriptive dominance in all four objective dimensions and displays two projections; it does not draw an unmeasured continuous frontier. Seed counts appear in the legend. All seven prespecified DPO weights must be retained.

- data/uf_objectives.csv: independent four-objective win rates, actual evaluation count and completed seeds.
- data/new_capability_results.csv: UF-4 main capability template and reported SafeRLHF IFEval aggregates.
- data/uf_tradeoffs.csv: full four-objective observed policy vectors and uncertainty.
- data/safe_three_seed_results.csv: supplied mean/spread, with unverified spread convention explicitly labeled.
- data/aggregation_policy_pairs.csv: supplied paired Nash-minus-utilitarian intervals.
- data/ifeval_per_seed.csv and refusal_truncation_audit.csv: arithmetic and denominator audit.

Historical CSVs preserve earlier one-seed snapshots. Do not copy pooled six-arm ranges into separate family means. The old IFEval source variant remains unverified and must be reconciled before final reporting. An empty value means unmeasured, not zero.

## Execute the revised campaign

CLAUDE_PROMPT.md contains the ready-to-use server instruction. REVISION_NOTES.md explains the evidence, original-paper dataset comparisons and early schedule. UF-4 is the new primary general-instruction panel, not an optional SafeRLHF extension. The plan uses a newly trained native-long-context four-head GPM/BT teacher, disjoint teacher/policy splits, shared new response pools, actual external-method adaptations and independent local policy evaluation.

Internal freezes are September 16 18:00 KST for training/model selection, September 18 18:00 for metrics, and September 20 for the paper. New paid judge API calls remain zero; released GPT-4 labels and local GPU costs are disclosed. Throughput estimates must be replaced by an actual 200-prompt profile.

## Provenance and limits

The new three-seed numbers are from the user's September 10 execution report; private remote artifacts were not independently recomputed here. Earlier controlled and teacher-validation numbers retain their existing supplied sources. This edit did not launch cluster training or verify the server's latest private commits.

main_v6.patch is the diff against the latest uploaded main_v6(1).tex. Integrate it with current server work rather than resetting the repository to an older public revision. The empty prospective tables/figure must be populated with real observations or removed before submission. This draft does not establish a Nash aggregation advantage, equivalence to base, or global LLM Pareto optimality.
