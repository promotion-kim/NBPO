# NBPO re-audit f0efbb5

Start with `NBPO_REAUDIT_f0efbb5.md` and `issues.json`. Read `code_evidence.html`
for original file lines and hashes. This package does not modify the repository.
It does not contain real model weights or new LLM benchmark results.

## Reproduce

Use a clean extracted checkout of commit
`f0efbb5935a4f522b8335292d6927161b274b796`, with NumPy/SciPy/PyTorch/PyYAML/pytest.
Run the new checks into a **fresh** output directory (fixtures are retained).

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
python reverify_f0efbb5.py --repo /absolute/path/NBPO-main --out-dir ./fresh_f0_checks

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
python reverify_97e03c0.py --repo /absolute/path/NBPO-main --out ./regression_results.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
python prior_regression_adapted.py --repo /absolute/path/NBPO-main --out ./prior_regression_results.json
```

`fixture_support.py` must be next to the scripts. The old reverify filename names
its origin, not the audited commit; `--repo` selects the actual source. Exit 0
means the audit harness completed, **not** that every finding is resolved.
Read individual PASS / OPEN / OBSERVATION entries. Repeated checks across panels
are not separate bug counts. The legacy global evaluator remains intentionally
global; its different decision is not evidence the local implementation ran it.

`new_checks_results.json` and raw CLI decision logs preserve Python JSON's NaN and
Infinity values for invalid-metric tests. `expanded_results.json` is the strict-JSON
copy: nonfinite values are labeled strings. Fixtures use synthetic comparisons,
logprobs and sentinel checkpoint files. No real model forward occurs.

The supplied repository selftest requires private `/work` artifacts, so this audit
uses its own portable fixtures and the **actual** local-gate CLI subprocess. To
run the new CLI, the tests add the helper snapshot already present in the checkout
to PYTHONPATH; the separate root/current-only import failure is also recorded.

For the repository pytest suite, reproduce the command in `pytest_cmd.json`,
replacing working directory and output paths. Actual result: 121 passed; five
failures due to absent `transformers`/`accelerate`. Exact IDs and causes are in
`pytest_summary.json`. These are environment-blocked, not classified code bugs.

Source ZIP identity, diff, unchanged integration paths, call-site search, source
integrity and package versions are recorded in separate JSON files. A source
revision is not proof a historical checkpoint used it or was re-evaluated with it.

## Note on this copy

The `fixtures/` tree this bundle generated is not committed: it is roughly 280
generated files, including binary `.npz`, and `reverify_f0efbb5.py --repo <repo>
--out-dir <dir>` rebuilds it. The report, the issue list, the harness and the
recorded results are here.
