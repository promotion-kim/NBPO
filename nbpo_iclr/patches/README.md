# Manuscript patches

Each file here is a **drop-in replacement for one named block** of
`main_v5.tex`, kept separate so the main file is not rewritten before the
corresponding result is final. Apply one at a time, and only the ones whose
results are frozen.

| patch | replaces | status of the result it reports |
|---|---|---|
| `section_5_2_and_algorithm_1.tex` | `\subsection{Sampled policy fitting and dual update}` (currently line 344) and the `algorithm` environment `alg:nbpo` (currently line 428) | **frozen** — controlled-v2 (`controlled_v2_manifest.json`) and the solver scaling sweep (`direct_solver_scaling_report.md`) |
| `appendix_rstep_ablation.tex` | **new** appendix subsection, referenced as `app:rstep-ablation` from the patch above | **frozen** — same sources |

## Labels the patches assume

Every `\ref` and `\eqref` in these patches resolves against `main_v5.tex`, and
`tests/test_manuscript_patches.py` fails if that stops being true — a patch that
introduces a dangling reference is worse than no patch.

Three labels I first wrote from memory did **not** exist and were repointed to
the real ones: `eq:centered-payoff` → `eq:centered`, `eq:dual-objective` →
`eq:prox-dual`, `eq:pairwise-target` → `eq:binary-target`. A fourth,
`sec:controlled`, does not exist because the controlled-nontransitivity section
is not written yet; the appendix patch points at `sec:experiments` and should be
repointed when that section lands.

`app:rstep-ablation` is defined by `appendix_rstep_ablation.tex` itself, so the
two patches must be applied together or neither.

## What the patches deliberately do not touch

The population theorem and every statement about the exact population proximal
update are unchanged. Only the description of how the *finite-pool* inner problem
is solved changes, plus the demotion of the alternating map to an ablation.

Nothing else in the manuscript is touched by these patches.
