"""Prove the self-contained main_v6.tex is the author's bytes with \\input removed.

main_v6.tex no longer matches protected_manifest.json, and it cannot: the
manifest hashes the \\input form, and the file is now self-contained by
request. Updating the manifest to make the check pass would throw away the
guarantee, so the check is replaced by a stronger one rather than weakened:

  1. every inlined block equals the current bytes of the file it names, and
  2. collapsing each block back to its \\input line reproduces
     provenance/main_v6_inputs.tex exactly, whose sha256 is the one the
     manifest declares for main_v6.tex.

If both hold, the manuscript is the protected manuscript plus indirection
removed, which is checkable at any later date. The other eleven protected
files are still checked by validate_protected.py untouched.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
TARGET = PAPER / "main_v6.tex"
SOURCE = PAPER / "provenance/main_v6_inputs.tex"
MANIFEST = PAPER / "protected_manifest.json"
BLOCKS = [("RESULT VALUES", "templates/results_auto.tex"),
          ("EXPERIMENT TEMPLATES", "templates/experiment_templates.tex")]


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main():
    text = TARGET.read_text(encoding="utf-8")
    findings, ok = [], True

    for name, rel in BLOCKS:
        pattern = re.compile(r"^%% BEGIN INLINE %s\b[^\n]*\n(.*?)^%% END INLINE %s[^\n]*\n"
                             % (re.escape(name), re.escape(name)),
                             re.DOTALL | re.MULTILINE)
        match = pattern.search(text)
        if not match:
            findings.append("block %s is missing" % name)
            ok = False
            continue
        current = (PAPER / rel).read_text(encoding="utf-8")
        if not current.endswith("\n"):
            current += "\n"
        if match.group(1) != current:
            findings.append("block %s does not match %s" % (name, rel))
            ok = False
        text = pattern.sub(lambda _m, rel=rel: "\\input{%s}\n" % rel, text, count=1)

    collapsed = text.encode("utf-8")
    declared = json.loads(MANIFEST.read_text())["main_v6.tex"]
    source_sha = sha_bytes(SOURCE.read_bytes()) if SOURCE.exists() else None
    collapsed_sha = sha_bytes(collapsed)

    if source_sha != declared:
        findings.append("provenance/main_v6_inputs.tex does not carry the declared hash")
        ok = False
    if collapsed_sha != declared:
        findings.append("collapsing the inlined file does not reproduce the declared hash")
        ok = False

    report = {"self_contained_sha256": sha_bytes(TARGET.read_bytes()),
              "collapsed_sha256": collapsed_sha,
              "declared_main_v6_sha256": declared,
              "input_form_sha256": source_sha,
              "equivalent": ok, "findings": findings}
    print(json.dumps(report, indent=1))
    if ok:
        print("main_v6.tex is the declared manuscript with \\input indirection removed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
