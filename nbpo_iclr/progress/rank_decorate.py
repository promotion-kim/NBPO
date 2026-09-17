"""Mark the best and second-best entry of each declared column group.

Table 1 and Table 2 compare methods down a column, so the reader needs the
ranking marked. The values live in \\UnionSet declarations, which a LaTeX macro
cannot rank for itself, so the decoration is applied to the stored value:
\\mathbf{} for the best and \\underline{} for the second best.

Three properties matter and are enforced:

  * idempotent -- existing decoration is stripped before ranking, so running
    this again after new cells arrive re-ranks rather than nests;
  * pending-safe -- an unmeasured cell takes no rank and does not occupy one, so
    a column with three measured cells marks a best and a second best among
    those three and says nothing about the rest;
  * tie-honest -- a tie for best bolds every row holding it and the next
    distinct value then becomes the second best; a tie for second underlines
    every row holding it.

Only the value inside a declaration changes. No row, column, caption or key is
added, removed or renamed.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

NUM = re.compile(r"-?\.?\d[\d.]*")


def read_declared(source: str, key: str):
    """(start, end, raw value) of one \\UnionSet declaration, brace-balanced."""
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
    return start, i, source[start:i]


def strip_decoration(value: str) -> str:
    """Undo a previous \\mathbf/\\underline wrap, leaving the plain value."""
    out = value
    for macro in ("\\mathbf", "\\underline"):
        pattern = re.compile(r"^\$%s\{(.*)\}\$$" % re.escape(macro))
        m = pattern.match(out)
        if m:
            out = "$%s$" % m.group(1)
    return out


def number_of(value: str):
    """The single plain number a cell holds, or None if it holds no such thing.

    Deliberately refuses anything it cannot order by reading one number:
    scientific notation like $9.6\\times10^{-11}$ would otherwise be compared on
    its mantissa alone and silently mis-rank a column. Tables 1 and 2 hold win
    fractions, MT scores and capability fractions, all plain; a column that is
    not should fail here rather than be ranked wrongly.
    """
    if "\\times" in value or "^" in value or "e-" in value or "e+" in value:
        raise SystemExit("%r is not a plain number; ranking it on its mantissa "
                         "would be wrong" % value)
    found = NUM.findall(value)
    if len(found) != 1:
        return None
    text = found[0]
    try:
        return float(text if not text.startswith(".") else "0" + text)
    except ValueError:
        return None


def decorate(source: str, keys, higher_is_better=True):
    """Bold the best and underline the second best among the measured keys."""
    plain, values = {}, {}
    for key in keys:
        start, end, raw = read_declared(source, key)
        value = strip_decoration(raw)
        plain[key] = value
        if value != "\\pending":
            n = number_of(value)
            if n is None:
                raise SystemExit("%s holds %r, which carries no number" % (key, value))
            values[key] = n
    order = sorted(set(values.values()), reverse=higher_is_better)
    best = order[0] if order else None
    second = order[1] if len(order) > 1 else None
    marks = {}
    for key, value in plain.items():
        if key in values and values[key] == best:
            marks[key] = "$\\mathbf{%s}$" % value.strip("$")
        elif key in values and second is not None and values[key] == second:
            marks[key] = "$\\underline{%s}$" % value.strip("$")
        else:
            marks[key] = value
    # rewrite back to front so earlier offsets stay valid
    spans = []
    for key in keys:
        start, end, _ = read_declared(source, key)
        spans.append((start, end, marks[key]))
    for start, end, value in sorted(spans, reverse=True):
        source = source[:start] + value + source[end:]
    return source, {"best": best, "second": second,
                    "measured": len(values), "of": len(keys)}


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text())
    tex = Path(spec.get("tex", "main_v8.tex"))
    source = tex.read_text()
    report = {}
    for group in spec["groups"]:
        source, info = decorate(source, group["keys"],
                                group.get("higher_is_better", True))
        report[group["name"]] = info
    tex.write_text(source)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
