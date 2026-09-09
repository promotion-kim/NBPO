# Fixed primary result collection

`collect_primary_results.py` reads only fixed paths named by the frozen
evaluation-chain plan, with exactly three identities: base, WBC-final1750 and
MSE-final1750. No optional control is assumed, and no latest/best-result glob or
fallback checkpoint is used. Frozen generation/evaluation/training files are
not edited. All consumed input files are SHA256-bound in the report ledger;
job commands, source archives, controller completion and full export-validation
reports bind the comparison to the fixed checkpoints and protocol.

Fixed score sources:

| Scope | Relative source namespace |
|---|---|
| IF/GSM all three identities | `evaluations/deterministic_primary_v1` |
| HarmBench all three | `evaluations/harmbench_primary_v1` |
| Skywork base self-tie check | `evaluations/skywork_base_v1` |
| Skywork candidate | `evaluations/skywork_{label}_v1` (label includes `_primary_v1`) |
| SafeRLHF all three | `evaluations/saferlhf_primary_v1/{label}` |
| XSTest null/unadjudicated | `evaluations/xstest_primary_unadjudicated_v1` |
| Finite-pool teacher RM | `teacher_rm/nash_repair_v2_v1/aggregate` |

The collector independently verifies raw response manifests, planned prompt
hashes/IDs, model hashes, raw terminal IDs/token lengths and matched decoding.
It recomputes scalar/micro means and seeded percentile intervals from per-item
records; paired deltas join by UID. IFEval instruction flags/counts are checked
against the source instruction inventory. The frozen GSM answer parser is
rerun on saved responses; parse failures stay incorrect in the EM denominator.
HarmBench keeps its classifier/hash routes and category denominators/failures.
Skywork uses exact scalar equality for half-credit ties, verifies the base's
self-comparison, and retains805/500/250 subgroup counts, failures and lengths.
These are local scalar-RM proxies, never official LC/judge win rates.

SafeRLHF V is independently recomputed from the saved calibrated3-seed,
2-objective,8-comparator probability arrays with beta0.25. Original d is
recomputed from the separate independent-reference learner8-by-comparator8
score tensors, then s=V-d is checked per prompt. Actual greedy-base V and its
paired delta remain separate. Both dev500/test1000 vectors and CIs are retained.
This is a deterministic greedy finite-game diagnostic, not Algorithm1 neural
policy acceptance or a globally untouched-test claim. Cached-base lineage and
probability hashes remain explicit. Teacher input truncation is not policy
output truncation. Finite-pool teacher/RM summaries are reaggregated from their
per-prompt records after validating shard/chunk/occurrence coverage and hashes.

XSTest remains450 planned, safe250/unsafe200, with null refusal/harm/appropriateness
labels. Its lack of adjudication is a declared protocol limitation, not a missing
numeric score to fill. Missing required artifacts or failed item coverage make
the report INCOMPLETE; changed hashes/means/CIs/counts make it INVALID.

Normal final collection (only after all fixed outputs exist):

```bash
CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=/work/nbpo_repair_20260909/code:/work/pylibs_eval \
python3 -m scripts.experiments.nbpo_repair_20260909.collect_primary_results \
  --root /work/nbpo_repair_20260909 \
  --output /work/nbpo_repair_20260909/reports/primary_results_v1 \
  --paper-fragments
```

By default, missing evidence gives a nonzero exit and no TeX. `--allow-incomplete`
permits an explicitly INCOMPLETE diagnostic JSON/Markdown report but cannot be
combined with `--paper-fragments`. INVALID evidence always gives nonzero exit.
Outputs are exclusive to a fresh report namespace: `primary_results.json`,
`primary_results.md`, manifest, and (COMPLETE only) `primary_main_table.tex` and
`primary_appendix.tex`. Neither fragment is inserted into the manuscript here.
The main table has two panels and the three authorized identities; the appendix
retains full intervals, denominators, vectors and diagnostic lengths/caps/keyword
counts. Keyword matches are never relabeled as semantic refusals.

CPU validation:23 synthetic contract tests exercise means/CIs, micro ratios,
parse-failure denominators, UID pairing, hash mutations, Unicode JSONL response
strings, reference arrays, the full dev500/test1000 vector schema, missing/invalid
states and publication-output gating. Synthetic test numbers are not results.
The real base-only collector dry-run is preserved as
`reports/collector_cpu_preflight_v2`: INCOMPLETE,35 missing sections, zero
validation errors, verified existing base Alpaca/Hard/Creative summaries, no
TeX. Its earlier v1 flagged a collector JSONL `splitlines()` bug on a valid
Unicode line separator; this was corrected to physical file-line iteration and
regression tested, not treated as source corruption. No GPU/model inference or
new evaluation is executed by collection.
