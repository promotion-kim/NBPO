"""Make main_v6.tex self-contained: no \\input except graphics.

The author's manuscript pulls two files in with \\input, and the paper is easier
to hand around as one file. This script writes that one file, and it is the
only thing allowed to write main_v6.tex:

    provenance/main_v6_inputs.tex   (the author's \\input form, protected bytes)
      + templates/results_auto.tex
      + templates/experiment_templates.tex
      -> main_v6.tex

Each pulled-in file lands between markers naming its source, so the block can
be refreshed without re-deriving the whole manuscript. That matters because
render_results.py keeps rewriting results_auto.tex as cells are measured: the
results path still ends in templates/results.json, and this script is what
carries the rendered block into the self-contained file.

Running it twice is the same as running it once. It works from the \\input form
or from an already-inlined file, so a refresh after a render does not need the
original at hand.

Content is not edited. The equivalence is checkable in the other direction:
progress/verify_inline_equivalence.py collapses the blocks back to their
\\input lines and compares the sha256 with protected_manifest.json, so the
manuscript can still be shown to be the author's bytes plus indirection
removed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
SOURCE = PAPER / "provenance/main_v6_inputs.tex"
TARGET = PAPER / "main_v6.tex"
# (marker name, path relative to the paper root)
BLOCKS = [("RESULT VALUES", "templates/results_auto.tex"),
          ("EXPERIMENT TEMPLATES", "templates/experiment_templates.tex")]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def begin(name, path):
    return ("%% BEGIN INLINE %s -- generated from %s by progress/inline_inputs.py; "
            "edit that file, never this block" % (name, path))


def end(name):
    return "%% END INLINE %s" % name


def block_pattern(name):
    return re.compile(r"^%% BEGIN INLINE %s\b.*?^%% END INLINE %s[^\n]*\n"
                      % (re.escape(name), re.escape(name)),
                      re.DOTALL | re.MULTILINE)


def render(source_text: str) -> str:
    out = source_text
    for name, rel in BLOCKS:
        content = (PAPER / rel).read_text(encoding="utf-8")
        if not content.endswith("\n"):
            content += "\n"
        replacement = "%s\n%s%s\n" % (begin(name, rel), content, end(name))
        input_line = re.compile(r"^\\input\{%s\}\n" % re.escape(rel), re.MULTILINE)
        existing = block_pattern(name)
        if input_line.search(out):
            out = input_line.sub(lambda _m: replacement, out, count=1)
        elif existing.search(out):
            out = existing.sub(lambda _m: replacement, out, count=1)
        else:
            raise SystemExit("found neither \\input{%s} nor an INLINE %s block"
                             % (rel, name))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report whether main_v6.tex is already up to date")
    args = ap.parse_args()

    if SOURCE.exists():
        source_text = SOURCE.read_text(encoding="utf-8")
    elif TARGET.exists():
        source_text = TARGET.read_text(encoding="utf-8")
    else:
        raise SystemExit("no source: expected %s" % SOURCE)

    new = render(source_text)
    old = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
    changed = new != old
    if args.check:
        print(json.dumps({"up_to_date": not changed, "target_sha256": sha(old),
                          "would_write_sha256": sha(new)}))
        return 0 if not changed else 1
    if changed:
        tmp = TARGET.with_suffix(".tex.tmp")
        tmp.write_text(new, encoding="utf-8")
        tmp.replace(TARGET)
    print(json.dumps({"wrote": str(TARGET), "changed": changed,
                      "inlined": [rel for _n, rel in BLOCKS],
                      "sha256": sha(new), "lines": new.count("\n")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
