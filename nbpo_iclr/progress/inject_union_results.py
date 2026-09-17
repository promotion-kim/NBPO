"""Populate the UNION RESULT CELLS declarations in main_v8.tex.

The appendix is explicit about the mechanism: "Allowed method IDs are visible in
the UNION RESULT CELLS block. The source uses \\UnionSet{key}{value}; populate
these declarations only, not unrelated text or historical values." So this does
not add a block of its own -- an added block would be overridden by the existing
declarations anyway, since TeX takes the last definition -- it rewrites the value
of each already-declared key that has been measured and leaves every other key
at \\pending.

Any key that is not already declared is an error: it means the collector invented
an ID the manuscript does not allow, and the run stops rather than inserting it.

Fractions print in the manuscript's leading-dot style; MT-Bench keeps its integer
part because it is a 1-10 score. Intervals, standard errors, sample sizes and
source paths stay in the artifacts rather than being squeezed into a cell.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path


def fmt(key: str, cell: dict) -> str:
    """One format per metric family, so a cell never misrepresents its units.

    Win fractions keep the manuscript's leading-dot style; MT-Bench keeps its
    integer part because it is a 1-10 score; a stage count is an integer; GPU
    hours get two decimals; and the two solver residuals are printed in
    scientific notation, because %.4f would render every certified residual as
    .0000 and make a 1e-10 solve indistinguishable from a 1e-14 one.
    """
    value = cell["value"]
    if key.endswith(".mt"):
        return "$%.2f$" % value
    if key.endswith(".stages"):
        if float(value) != int(value):
            raise SystemExit("%s is a stage count, not %r" % (key, value))
        return "$%d$" % int(value)
    if key.endswith(".gpu_h"):
        return "$%.2f$" % value
    if key.endswith((".inner_residual", ".dual_residual")):
        if value <= 0:
            raise SystemExit("%s must be a positive residual, got %r" % (key, value))
        power = math.floor(math.log10(value))
        mantissa = value / (10.0 ** power)
        if round(mantissa, 1) >= 10.0:      # 9.996e-11 is 1.0e-10, not 10.0e-11
            mantissa, power = mantissa / 10.0, power + 1
        return "$%.1f\\!\\times\\!10^{%d}$" % (mantissa, power)
    text = "%.4f" % value
    return "$%s$" % (text[1:] if text.startswith("0.") else text)


def replace_declared(source: str, key: str, value: str) -> str:
    """Rewrite one \\UnionSet declaration, counting braces rather than guessing.

    An earlier version matched the value as [^}]*, which is correct only while
    every value is brace-free. Once a cell holds scientific notation such as
    $9.6\\times10^{-11}$ that pattern stops at the inner brace, and re-injecting
    the same key would leave the tail behind and produce broken LaTeX. This
    scans to the matching close brace instead, so a cell can be rewritten as
    many times as new measurements arrive.
    """
    opener = "\\UnionSet{%s}{" % key
    hits = [i for i in range(len(source)) if source.startswith(opener, i)]
    if len(hits) != 1:
        raise SystemExit("%s is declared %d times" % (key, len(hits)))
    start = hits[0] + len(opener)
    depth, i = 1, start
    while i < len(source):
        c = source[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    if depth != 0:
        raise SystemExit("%s has an unbalanced declaration" % key)
    return source[:start] + value + source[i:]


def main() -> int:
    cells_path = Path(sys.argv[1])
    tex = Path("main_v8.tex")
    cells = json.loads(cells_path.read_text())["cells"]
    source = tex.read_text()

    # an earlier version of this script appended its own block; remove it so the
    # declared block stays the single source of truth
    stale = re.search(r"% BEGIN UNION RESULTS.*?% END UNION RESULTS\n", source, re.S)
    if stale:
        source = source[:stale.start()] + source[stale.end():]

    missing = [k for k in cells
               if not re.search(r"\\UnionSet\{%s\}\{" % re.escape(k), source)]
    if missing:
        raise SystemExit("keys not declared in the manuscript: %s" % sorted(missing))

    filled = {}
    for key, cell in sorted(cells.items()):
        value = fmt(key, cell)
        source = replace_declared(source, key, value)
        filled[key] = value
    tex.write_text(source)
    remaining = len(re.findall(r"\\UnionSet\{[^}]*\}\{\\pending\}", source))
    print(json.dumps({"populated": len(filled), "still_pending": remaining,
                      "values": filled}, indent=1)[:900])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
