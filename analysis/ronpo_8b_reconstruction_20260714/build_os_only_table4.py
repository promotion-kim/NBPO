#!/usr/bin/env python3
"""Generate the conditional fresh RONPO-OS Table-4 fragment from measured JSON/CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LABELS = {
    "base": "Base", "ronpo_os": "RONPO (OS)", "ronpo_full_expect": "RONPO (full-exp.)",
    "ronpo_k_only": "RONPO (top-mass)", "ipo": "IPO", "simpo": "SimPO",
    "sppo_avg": "SPPO (avg)", "inpo_avg": "INPO (avg)",
    "ht_mnpo_helpfulness": "HT-MNPO (help.)", "ht_mnpo_safety": "HT-MNPO (safety)",
    "ht_mnpo_conciseness": "HT-MNPO (concise)",
}


def f(value: float) -> str:
    return f"{float(value):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--local-summary", type=Path, required=True)
    parser.add_argument("--local-objectives", type=Path, required=True)
    parser.add_argument("--panel-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    if decision.get("decision") not in {"PASS", "PARTIAL"}:
        raise RuntimeError("Table 4 generation is forbidden by the precommitted FAIL decision")
    local = json.loads(args.local_summary.read_text(encoding="utf-8"))
    panel = json.loads(args.panel_summary.read_text(encoding="utf-8"))
    with args.local_objectives.open(encoding="utf-8") as handle:
        objective_rows = list(csv.DictReader(handle))
    objectives = {(row["model"], row["objective"]): row for row in objective_rows}
    panel_by = {row["model"]: row for row in panel["ranked"]}
    ranked = local["ranked_all_eligible_candidates"]
    best = ranked[0]["model"]
    lines = [
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Method & Worst (95\% CI) & Skywork & Athene & ArmoRM & Disp. & Panel worst (95\% CI) \\",
        r"\midrule",
    ]
    for row in ranked:
        model = row["model"]
        label = LABELS.get(model, model.replace("_", r"\_"))
        if model == "ronpo_os" and model == best:
            label = r"\textbf{" + label + "}"
        worst = f(row["worst_objective_marginal_win_rate"])
        wci = row["worst_objective_marginal_win_rate_ci95"]
        values = [f(float(objectives[(model, obj)]["marginal_win_rate_vs_base"]))
                  for obj in ("skywork", "athene", "armo")]
        disparity = f(row["cross_objective_marginal_spread"])
        if model in panel_by:
            prow = panel_by[model]
            panel_value = f(prow["worst_objective_marginal"])
            pci = prow["worst_objective_marginal_ci95"]
            panel_text = f"{panel_value} [{f(pci[0])}, {f(pci[1])}]"
        else:
            panel_text = "--"
        lines.append(f"{label} & {worst} [{f(wci[0])}, {f(wci[1])}] & "
                     f"{values[0]} & {values[1]} & {values[2]} & {disparity} & {panel_text} " + r"\\")
    failed = sorted(set(local.get("stability_failed_models", [])))
    if failed:
        escaped = ", ".join(value.replace("_", r"\_") for value in failed)
        lines.extend([r"\midrule", rf"\multicolumn{{7}}{{l}}{{Stability failed: {escaped}}} \\"])
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
