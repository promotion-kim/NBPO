#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

HEADS = ("instruction_following", "truthfulness", "honesty", "helpfulness", "safety")


def f(x):
    return f"{x:+.6f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    d = json.loads(a.summary.read_text())
    primary = d["comparisons"]["primary_all_objective_envelope"]
    verdict = "PASS" if primary["pass"] else "FAIL"
    lines = [
        "# UF5 Stage-6 joint baseline comparison",
        "",
        f"Primary verdict: **{verdict}**.",
        "",
        "All nine policies were scored in one ArmoRM BF16 batch context on the frozen 586-prompt panel.",
        f"The minimum Stage-6 margin over the per-head non-RONPO envelope was {f(primary['estimate'])} "
        f"(95% CI [{f(primary['ci95'][0])}, {f(primary['ci95'][1])}]).",
        "",
        "## Margin over the non-RONPO envelope",
        "",
        "| Head | Stage-6 margin |",
        "|---|---:|",
    ]
    lines += [f"| {h} | {f(primary['margin_by_objective'][h])} |" for h in HEADS]
    lines += ["", "## Pairwise comparisons", ""]
    for label in ("Base", "RMOD K=16", "INPO", "SPPO", "DPO", "IPO", "SimPO"):
        c = d["comparisons"][f"RONPO-COMB-S6_minus_{label}"]
        r = c["robust_floor_delta"]
        lines.append(
            f"- Versus {label}: robust-floor delta {f(r['estimate'])}, "
            f"95% CI [{f(r['ci95'][0])}, {f(r['ci95'][1])}]; "
            f"average delta {f(c['average_delta'])}."
        )
    lines += [
        "",
        "## Interpretation",
        "",
        ("Stage 6 clears the preregistered all-objective superiority gate."
         if primary["pass"] else
         "Stage 6 does not clear the preregistered all-objective superiority gate. "
         "The strong claim that it improves every objective over the strongest baseline is not supported."),
        "",
        "No model, head, prompt, or metric was selected after scoring.",
    ]
    a.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

