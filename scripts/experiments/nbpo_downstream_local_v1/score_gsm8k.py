#!/usr/bin/env python3
"""GSM8K zero-shot CoT exact match under the prompt this protocol pins.

The generation prompt ends with `End with "The answer is <number>"`, so the
parser looks for that form first. A response with no number in the specified
format counts as incorrect -- an unparseable answer is a wrong answer, not a
missing datum, and treating it otherwise would let a model that stops answering
look better.

Reported as `GSM8K zero-shot-CoT EM (specified prompt)`. It is not comparable to
published 8-shot numbers.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SPEC = re.compile(r"[Tt]he answer is\s*\$?(-?[\d,]+(?:\.\d+)?)")
ANYNUM = re.compile(r"(-?[\d,]+(?:\.\d+)?)")


def norm(x):
    try:
        v = float(str(x).replace(",", "").rstrip("."))
        return int(v) if v == int(v) else v
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.responses).open() if l.strip()]
    n = correct = unparsed = 0
    per = []
    for r in rows:
        gold_raw = r["meta"]["gold"].split("####")[-1].strip()
        gold = norm(gold_raw)
        m = SPEC.findall(r["output"])
        pred = norm(m[-1]) if m else None
        if pred is None:
            unparsed += 1
        ok = pred is not None and gold is not None and pred == gold
        correct += ok
        n += 1
        per.append({"uid": r["uid"], "gold": gold, "pred": pred, "correct": bool(ok),
                    "in_specified_format": bool(m)})
    out = {"label": args.label, "metric": "GSM8K zero-shot-CoT EM (specified prompt)",
           "n": n, "correct": correct, "accuracy": correct / n if n else None,
           "no_answer_in_specified_format": unparsed,
           "note": ("responses without the pinned answer format are scored incorrect; "
                    "not comparable to published 8-shot GSM8K numbers")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": out, "per_item": per}, indent=2) + "\n")
    print(f"{args.label:>12}  GSM8K EM={out['accuracy']:.4f}  n={n}  "
          f"unformatted={unparsed}")


if __name__ == "__main__":
    main()
