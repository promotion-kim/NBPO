# b949919 follow-up audit bundle

Read `NBPO_REAUDIT_b949919.md` first. `issues.json` contains actionable per-issue acceptance criteria.

Files:
- `diff_inventory.json`, `modified_files.diff`: byte-derived archive comparison.
- `audit_manifest.json`, `integrity.json`: target identity, environment and source integrity.
- `code_evidence.html` / `.md`: current source snippets with exact paths, line numbers, SHA256.
- `prior_audit_checks.py`: unchanged prior diagnostic program; named bug checks may record `REPRODUCED_FAILURE` without making the audit process crash.
- `prior_suite_new_results.json`, `prior_suite_new.log`: actual re-execution against b949919, not copied old results.
- `reverify_b949919.py`, `expanded_results.json`, `expanded_checks.log`: new path-specific CPU checks.
- `pytest_subset.log`: actual repository subset execution log.

Run from an environment with NumPy, SciPy, torch and pytest:

```bash
python prior_audit_checks.py --repo /absolute/path/NBPO-main --out repeated.json
python reverify_b949919.py --repo /absolute/path/NBPO-main --out expanded.json
```

Some source functions are executed by extracting their original AST, to inspect their calculations without unavailable heavy imports. This is marked in each result and is NOT a substitute for a full import or end-to-end test. No audited source is modified, no model weights are downloaded and no LLM inference is run.

Repository subset command (same 5 optional-dependency checks excluded as in prior audit):

```bash
cd /absolute/path/NBPO-main
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD" \
python -m pytest -q tests/test_nbpo_core.py tests/test_nbpo_dual.py \
  tests/test_nbpo_targets.py tests/test_nbpo_provenance.py \
  -k 'not test_dataset_splits_match_saved_dataset and not test_run_config_carries_expected_artifact_hashes and not test_stale_expected_artifact_hashes'
```
