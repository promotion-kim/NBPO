# NBPO re-audit 97e03c0

Read `NBPO_REAUDIT_97e03c0.md` first. `issues.json` distinguishes resolved issues, the remaining Algorithm-1 integration gap, and input-integrity hardening. This archive contains no model weights or source-code patch.

## Reproduce

Place this folder and an extracted 97e03c0 checkout locally, with NumPy/SciPy/PyTorch/PyYAML available.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
python reverify_97e03c0.py --repo /absolute/path/NBPO-main --out expanded_results.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
python prior_regression_adapted.py --repo /absolute/path/NBPO-main --out prior_regression_results.json
```

The first script needs `fixture_support.py` beside it. Exit 0 means all audit checks ran without harness error, **not** that every audit finding is closed. Read each PASS/OPEN status and scope. Synthetic fixtures test invariants; they are not LLM performance measurements.

`code_evidence.html` contains original source lines and hashes. `modified_files.diff` records the archive-to-archive diff. `audit_manifest.json` records archive identifiers/environment. `source_integrity.json` confirms the original tracked bytes are unchanged.

The prior harness was updated only for the newly implemented API signature and the added `math` dependency in the isolated legacy gate function. No audited source was modified.
