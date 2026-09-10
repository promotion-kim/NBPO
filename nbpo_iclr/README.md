# NBPO v6: response to the 34 professor/GPT review comments

This revision applies the comments in the newly supplied older main.tex to the current 8B-main-experiment v6. It preserves the Nash objective, exact population theorem, current result CSVs and general-instruction experiment plan. Read REVIEW_RESPONSE.md for every review's meaning, resolution and remaining evidence.

## Files and Overleaf

Set the Overleaf Main document to main_v6.tex. Upload this complete directory so generated/, figures/, data/, style and bibliography dependencies remain available. main_v6.patch compares with the immediately preceding v6; main_v6_vs_reviewed_source.patch compares with the older annotated main.tex. The annotated source is an input, not the revised manuscript.

```bash
python3 scripts/render_tables.py
python3 scripts/plot_uf_tradeoffs.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main_v6.tex
```

The compiled manuscript is 27 pages, with the conclusion on page 9. No undefined-reference/citation or overfull warnings remain. A harmless LaTeX float-placement warning changes h to ht. All pages were inspected through a contact sheet and the changed main/math appendix pages at reading resolution.

## Changes

- Move the regression square inside the expectation. Available public e90f2a code already squares each example before averaging; this is a verified manuscript defect, not proof that the current server trainer is wrong.
- Explain and independently reproduce instability of the old undamped policy/opponent map; give the direct finite-pool objective's gradient and strictly negative Hessian.
- Separate population, finite-pool and neural policies; shared dual weights remain global across prompts. Refresh learner pools within each outer stage.
- State failure/feasibility status and unprojected stationarity checks without restoring a separate neural warm-start stage.
- Qualify BT/cycle motivation, equal bargaining weight, game value versus direct wins, primitive scaling, aggregation order, finite-support identification and exact-solve convergence.
- Correct MOPO's preference-only classification, discuss RACO, and update Zhong et al.'s September 2026 v2 metadata.
- Retain completed SafeRLHF null results and pending 8B objective/capability templates. Add a small independent cross-play/common-evaluation protocol; do not turn every review into a new full campaign.

REVISION_NOTES.md and REVIEW_RESPONSE.md contain the review response. EXPERIMENT_PLAN.md preserves the prior early experiment plan with the MOPO classification corrected. CLAUDE_PROMPT.md retains that campaign and appends targeted review checks. Core result CSV bytes are unchanged.

## Reproduce the finite-game checks

```bash
python3 scripts/verify_direct_solver_counterexamples.py data/verify_direct_solver_counterexamples.json
```

Requires NumPy and SciPy. The deterministic R06/R12 calculations are not LLM results. They fix the multiplier for an inner solve; they do not pretend to solve the full Nash dual or establish a neural convergence theorem. Original extracted comments are in review_sources/ for traceability.

Remote cluster training was not executed in this manuscript edit, and the newest private run artifacts were not independently verified. UF-4 cells and empty figure remain prospective. Before submission, populate them from real matched artifacts or remove them. See data/MAIN_EXPERIMENTS_SCHEMA.md for source and metric definitions.
