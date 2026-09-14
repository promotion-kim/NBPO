"""Merge measured cells into templates/results.json, then render and validate.

The protected manuscript reaches new results only through this path:

    campaign artifact on the pod
      -> a bundle file {contract_id, artifact, sha256, cells:{key: record}}
      -> templates/results.json
      -> render_results.py
      -> templates/results_auto.tex

Every record is checked before it is written, because the renderer trusts what
it is given: the key must be one of the declared 81, the value must be a finite
number, the artifact path must be non-empty, and the sha256 must be 64 hex
characters. A cell that fails any check is refused and the file is left
untouched, so a half-written table cannot reach a build.

Cells already measured are not silently overwritten: a rewrite has to pass
--replace, and the previous record is kept in the file's history list with the
reason. Nothing here deletes a declared key or turns an unmeasured cell into 0.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
RESULTS = PAPER / "templates/results.json"
KEYS = PAPER / "templates/result_keys.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def fail(msg):
    print("REFUSED: " + msg, file=sys.stderr)
    raise SystemExit(2)


def check(key, record, declared):
    if key not in declared:
        fail("key %r is not one of the %d declared keys" % (key, len(declared)))
    if not isinstance(record, dict):
        fail("record for %s must be an object" % key)
    value = record.get("value")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        fail("value for %s must be numeric, got %r" % (key, value))
    if value != value or value in (float("inf"), float("-inf")):
        fail("value for %s must be finite" % key)
    artifact = record.get("artifact")
    if not isinstance(artifact, str) or not artifact.strip():
        fail("artifact path for %s is missing" % key)
    sha = record.get("sha256")
    if not isinstance(sha, str) or not HEX64.match(sha):
        fail("sha256 for %s must be 64 hex characters, got %r" % (key, sha))
    ci = record.get("ci95")
    if ci is not None:
        if not (isinstance(ci, list) and len(ci) == 2
                and all(isinstance(x, (int, float)) for x in ci)):
            fail("ci95 for %s must be a pair of numbers" % key)
        if not (ci[0] <= value <= ci[1]):
            fail("value %r for %s lies outside its own ci95 %r" % (value, key, ci))
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True,
                    help="json with contract_id, artifact, sha256 and cells")
    ap.add_argument("--replace", action="store_true",
                    help="allow overwriting cells that already carry a record")
    ap.add_argument("--reason", default="", help="why a replacement is correct")
    ap.add_argument("--render", action="store_true", help="run render_results.py after")
    args = ap.parse_args()

    declared = set(json.loads(KEYS.read_text()))
    doc = json.loads(RESULTS.read_text())
    bundle = json.loads(Path(args.bundle).read_text())
    cells = bundle.get("cells") or {}
    if not cells:
        fail("the bundle carries no cells")

    artifact = bundle.get("artifact")
    sha = bundle.get("sha256")
    contract = bundle.get("contract_id")
    if not contract:
        fail("the bundle has no contract_id")

    prepared, replacing = {}, []
    for key, record in cells.items():
        if not isinstance(record, dict):
            record = {"value": record}
        record.setdefault("artifact", artifact)
        record.setdefault("sha256", sha)
        record.setdefault("contract_id", contract)
        record.setdefault("measured_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                        time.gmtime()))
        prepared[key] = check(key, record, declared)
        if doc["cells"].get(key) is not None:
            replacing.append(key)

    if replacing and not args.replace:
        fail("these cells already carry a record and --replace was not given: %s"
             % ", ".join(sorted(replacing)))
    if replacing and not args.reason:
        fail("a replacement needs --reason")

    history = doc.setdefault("replaced_records", [])
    for key in replacing:
        history.append({"key": key, "previous": doc["cells"][key],
                        "reason": args.reason,
                        "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    doc["contract_id"] = doc.get("contract_id") or contract
    doc["cells"].update(prepared)
    measured = sum(1 for v in doc["cells"].values() if v is not None)
    tmp = RESULTS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    tmp.replace(RESULTS)
    print(json.dumps({"written": sorted(prepared), "replaced": sorted(replacing),
                      "measured_cells": measured, "declared_cells": len(declared)}))

    if args.render:
        for step in (["python3", "validate_protected.py"],
                     ["python3", "render_results.py"],
                     ["python3", "validate_protected.py"]):
            r = subprocess.run(step, cwd=PAPER, capture_output=True, text=True)
            print("  %-24s %s" % (" ".join(step[1:]), (r.stdout or r.stderr).strip()[:160]))
            if r.returncode != 0:
                fail("%s exited %d" % (step[-1], r.returncode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
