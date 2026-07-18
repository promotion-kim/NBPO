# Paper update audit

- Paper target: `ronpo_aaai/main.tex`, table `tab:saferlhf-robust` and its directly associated Results and Limitations text.
- Seed inputs: `inputs/seed42_ranked_validation_summary.json` and `inputs/seed43_ranked_validation_summary.json`.
- Prompt manifest: `inputs/fresh_default_test_1000.jsonl`, SHA-256 `c7b5d42f5b866d6c8fce8667cfb22d27541fa86eea61d4b9d2ad0dad7a12eec2`.
- Aggregation: arithmetic mean and sample standard deviation across training seeds 42 and 43 after each seed is independently normalized on the same prompt and method pool.
- Paper handling: RONPO top-mass remains an estimator ablation and is omitted from Table 4; its measurements remain in `FULL_TWO_SEED_METRICS.md`.
- Build: TinyTeX `pdflatex`, `bibtex`, and two final `pdflatex` passes completed with no fatal error and no overfull box. The pre-existing unresolved bibliography keys are `son2026rmod` and `zhong2024panacea`.
- Generated paper SHA-256: `main.tex` `305155dc4411bc77e0371a211df1eaffaacc1eab2e23dedd81fdebe044251c3f`; `main.pdf` `5fbd257a8e338ec9d5660ced46866982f437b43b41216a920c3296f0e967a9cd`.
