# Nash Bargaining Preference Optimization (NBPO)

Source package for the ICLR 2027 submission.

NBPO separates two decisions in multi-objective alignment. An objective-wise preference game
summarises potentially cyclic pairwise feedback into one game value per objective, and Nash
bargaining selects a single compromise across those values, balancing improvements over a
reference fallback rather than over prespecified weights.

## Contents

| Path | Description |
|---|---|
| `main_v2.tex`, `main_v2.pdf` | Current manuscript (25 pages). |
| `main.tex`, `main.pdf` | Previous revision (35 pages), kept for reference. |
| `iclr2027_conference.{sty,bst,bib}` | Style files and bibliography database. |
| `math_commands.tex` | Math macros used by the manuscript. |
| `figures/` | Vector figures. |
| `finite_pool_certificate_table.tex`, `generated_simplification.tex` | Generated table fragments included by the manuscript. |
| `REVISION_NOTES.md`, `THEOREM_AUDIT.md`, `EXPERIMENTS_REQUIRED.md` | Working notes on methodological changes, theorem scope, and outstanding experiments. |

## Build

```bash
pdflatex main_v2
bibtex   main_v2
pdflatex main_v2
pdflatex main_v2
```

The `.bst` is an ICLR-compatible fallback; the uploaded style bundle did not include a 2027
bibliography style. Replace it with the official file before submission if one is distributed.

## Codebase

The training and evaluation code is taken from the
[MNPO](https://github.com/smiles724/MNPO) pipeline in the parent repository (`mnpo_scripts/`,
`alignment/`, `on_policy_data_gen/`, `accelerate_configs/`). NBPO is implemented within that
pipeline rather than as a separate stack: the constrained proximal update reduces to the existing
regression trainer with the bargaining weights supplied as per-pair targets, so the only additions
are the dual solve that produces those weights and the pair builder that applies them.

## Status

Results are reported on prompts audited to be disjoint from each arm's training pairs and
reference pool; see the prompt-overlap appendix for the audit and for which panels were rescored.
Evidence supports the Nash aggregation over utilitarian and max-min alternatives. It does not
currently support the adaptive game-value representation over a fixed-reference margin, and the
manuscript says so.
