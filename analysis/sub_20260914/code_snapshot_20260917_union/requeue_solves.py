"""Re-dispatch the six solves after fixing the SyntaxError in the solver copies.

A FAILED or BLOCKED job re-runs only when its spec_sha256 changes, so each spec
records why it is being retried. Nothing else about the specs changes.
"""
import json
from pathlib import Path

Q = Path("/work/uf4_20260910/jobs/queue")
touched = []
for p in sorted(Q.glob("*.json")):
    spec = json.loads(p.read_text())
    if not spec["job_id"].startswith("sub_solve_"):
        continue
    spec["retry_note"] = ("2026-09-14 21:10 KST: the first attempt died with a "
                          "SyntaxError because the generated solver copies prepended a "
                          "second module docstring, which pushed `from __future__ import "
                          "annotations` out of first position. The copies now carry their "
                          "provenance as comments; the solver code itself is unchanged.")
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(spec, indent=2) + "\n")
    tmp.replace(p)
    touched.append(spec["job_id"])
print(json.dumps({"re_dispatched": touched}, indent=1))
