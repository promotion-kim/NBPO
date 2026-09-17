"""One protection verdict for the tooling, now that main_v6.tex is self-contained.

validate_protected.py hashes whole files, so it reports main_v6.tex as changed
the moment the \\input indirection is removed. That single expected difference
must not read as a protection failure, and every other difference still must.

status() therefore accepts exactly one deviation, and only when it is proved
harmless:

  * all twelve protected files unchanged  -> ok
  * only main_v6.tex differs AND progress/verify_inline_equivalence.py passes,
    i.e. collapsing its inlined blocks reproduces the declared bytes -> ok
  * anything else -> not ok, with the file list

The manifest is never rewritten to make a check pass.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]


def status():
    """(ok, one-line message) for the whole protection contract."""
    r = subprocess.run([sys.executable, "validate_protected.py"], cwd=PAPER,
                       capture_output=True, text=True)
    if r.returncode == 0:
        return True, "all protected files unchanged"
    changed = [line.strip() for line in r.stdout.splitlines()[1:] if line.strip()]
    if changed != ["main_v6.tex"]:
        return False, "protected files changed: " + ", ".join(changed or ["unknown"])
    e = subprocess.run([sys.executable, "progress/verify_inline_equivalence.py"],
                       cwd=PAPER, capture_output=True, text=True)
    if e.returncode == 0:
        return True, ("main_v6.tex is self-contained and collapses to the declared "
                      "bytes; the other eleven files are unchanged")
    tail = (e.stdout or e.stderr).strip().splitlines()
    return False, "main_v6.tex is not equivalent to the declared manuscript: %s" % (
        tail[-1][:120] if tail else "equivalence check failed")


if __name__ == "__main__":
    ok, message = status()
    print(("PASS " if ok else "FAIL ") + message)
    sys.exit(0 if ok else 1)
