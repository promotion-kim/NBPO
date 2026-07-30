#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

HEADS = ("instruction_following", "truthfulness", "honesty", "helpfulness", "safety")


def fmt(x):
    return f"{x:+.6f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    d = json.loads(a.summary.read_text())
    x = d["primary"]
    verdict = "PASS" if x["pass"] else "FAIL"
    lines = [
        "# Average warm start followed by robust fine-tuning",
        "",
        f"Primary verdict: **{verdict}**.",
        "",
        "The two arms share the INPO Stage-1 initialization, response pool, prompt rows, optimizer, "
        "step count, seed, and checkpoint rule. They differ only in pair construction and loss.",
        "",
        f"The minimum five-head RONPO-minus-INPO continuation margin is {fmt(x['minimum_head_delta'])} "
        f"(95% CI [{fmt(x['minimum_head_delta_ci95'][0])}, "
        f"{fmt(x['minimum_head_delta_ci95'][1])}]).",
        "",
        "| Head | Paired delta | 95% CI |",
        "|---|---:|---:|",
    ]
    for h in HEADS:
        lo, hi = x["paired_delta_ci95"][h]
        lines.append(f"| {h} | {fmt(x['paired_delta_by_objective'][h])} | [{fmt(lo)}, {fmt(hi)}] |")
    lines += [
        "",
        ("The preregistered all-objective matched-control gate passes."
         if x["pass"] else
         "The preregistered all-objective matched-control gate does not pass."),
        "",
        "All outcomes are reported without reward-based checkpoint selection or retry.",
    ]
    a.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

