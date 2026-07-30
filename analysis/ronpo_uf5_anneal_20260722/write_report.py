#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument("--summary", type=Path, required=True); p.add_argument("--out", type=Path, required=True); p.add_argument("--root", type=Path); a = p.parse_args()
    d = json.loads(a.summary.read_text()); methods = d["methods"]
    lines = ["# RONPO UF5 annealing experiment", "", "## Protocol", "", f"All models were scored together on the locked {d['protocol']['n_prompts']}-prompt intersection. The primary endpoint is the paired worst-head delta from Base with a 2,000-resample seed-42 prompt bootstrap interval. No stage or arm was selected by reward.", "", "## Results", "", "| Arm | Stage | Worst head | Worst reward | Delta vs Base | 95% CI | Verdict |", "|---|---:|---|---:|---:|---:|---|"]
    for arm, prefix in [("Moving anchor", "RONPO-MA-S"), ("Stronger signal", "RONPO-SS-S")]:
        found = False
        for stage in range(1, 5):
            label = f"{prefix}{stage}"
            if label not in methods:
                key = "moving_anchor" if prefix == "RONPO-MA-S" else "stronger_signal"
                gate = a.root / key / f"stage{stage}" / "eval" / "stability_gate.json" if a.root else None
                if gate and gate.is_file() and not json.loads(gate.read_text()).get("passed", False):
                    status = "stability gate failed"
                elif a.root and any((a.root / key / f"stage{s}" / "eval" / "GATE_FAILED").is_file() for s in range(1, stage)):
                    status = "not run after earlier fail-closed gate"
                else:
                    status = "missing or unfinished"
                lines.append(f"| {arm} | {stage} | - | - | - | - | {status} |")
                continue
            found = True; r = methods[label]; lo, hi = r["worst_head_paired_delta_ci95"]
            verdict = "separates above Base" if lo > 0 else ("separates below Base" if hi < 0 else "does not separate")
            lines.append(f"| {arm} | {stage} | {r['worst_objective']} | {r['worst_objective_mean']:.6f} | {r['worst_head_paired_delta_vs_base']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {verdict} |")
        vals = [methods[f"{prefix}{s}"] for s in range(1,5) if f"{prefix}{s}" in methods]
        separating = [s for s in range(1,5) if f"{prefix}{s}" in methods and methods[f"{prefix}{s}"]["worst_head_paired_delta_ci95"][0] > 0]
        verdict = f"Stages {separating} separate above Base on the preregistered primary endpoint." if separating else "No completed stage separates above Base on the preregistered primary endpoint."
        lines += ["", f"**{arm} verdict:** {verdict}", ""]
    lines += ["## Interpretation", "", "The table reports the locked result without checkpoint selection or retries based on reward. Per-head and average paired effects are retained in `paired_summary.json` and `paired_means.csv`.", ""]
    a.out.write_text("\n".join(lines))


if __name__ == "__main__": main()
