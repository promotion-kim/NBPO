#!/usr/bin/env python3
"""Generate the canonical result CSVs and TeX from the run registry.

Every number in the manuscript has to be traceable to a run that actually
finished, so the tables are built from the registry and the artifacts it points
at -- never typed in, and never carried over from a previous campaign.

The rule this file exists to enforce: **an incomplete cell is `pending`, not a
number.** A row is only eligible to be reported when all three final seeds
completed and were evaluated; anything less is emitted as `\\pending{}` in the
TeX and left empty in the CSV, with the reason recorded. There is no code path
here that fills a cell from fewer seeds, from the best seed, from a stale
artifact, or from a default.

Row order is fixed by the protocol and does not depend on the results.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

TABLE1_ROWS = [
    ("base", "Base ($\\pi_\\mathrm{ref}$)"),
    ("nbpo", "NBPO"),
    ("fixed-reference-nash", "Fixed-reference Nash"),
    ("bt-rm-nash", "BT-RM-Nash"),
    ("game-utilitarian", "Game-utilitarian"),
    ("game-ks", "Game-KS"),
    ("bt-rm-utilitarian", "BT-RM-utilitarian"),
    ("prosper", "PROSPER / MaxEntBW"),
    ("mopo", "MOPO"),
    ("raco", "RACO"),
]
FINAL_SEEDS = (42, 43, 44)
OBJECTIVES = ["instruction_following", "truthfulness", "honesty", "helpfulness"]
BENCHMARKS = ["arena_hard_v2", "alpaca_eval_2", "mt_bench", "ifeval", "truthfulqa"]


def read_jsonl(p: Path):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def load_registry(root: Path) -> dict:
    runs = {}
    for rec in read_jsonl(root / "run_registry.jsonl"):
        runs[rec["run_id"]] = rec           # last write wins
    return runs


def mean_std(vals):
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], None
    return statistics.mean(vals), statistics.stdev(vals)


def cell(mean, std, complete):
    """A value only renders when the row is complete. Otherwise: pending."""
    if not complete or mean is None:
        return "\\pending{}"
    return f"{mean:.3f}" + (f" $\\pm$ {std:.3f}" if std is not None else "")


def collect(root: Path, registry: dict):
    """Per-method, per-seed evaluation results, from whatever has completed."""
    out = {}
    for method, _ in TABLE1_ROWS:
        seeds = {}
        for seed in FINAL_SEEDS:
            path = root / "eval" / f"{method}_s{seed}" / "objective_win_rates.json"
            run = registry.get(f"eval-{method}-s{seed}")
            if not path.exists():
                continue
            if run is not None and run.get("status") != "complete":
                continue                     # an artifact from a run that did not finish
            seeds[seed] = json.loads(path.read_text())
        missing = [s for s in FINAL_SEEDS if s not in seeds]
        out[method] = {"seeds": seeds, "complete": not missing, "missing_seeds": missing}
    return out


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    registry = load_registry(args.root)
    data = collect(args.root, registry)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # --- per-seed -----------------------------------------------------------
    rows = []
    for method, label in TABLE1_ROWS:
        for seed, res in sorted(data[method]["seeds"].items()):
            rows.append([method, label, seed,
                         res.get("worst_objective"), res.get("average_objective"),
                         *[res.get("objectives", {}).get(o) for o in OBJECTIVES],
                         res.get("mean_output_length"), res.get("policy_reference_kl")])
    write_csv(out / "table1_per_seed.csv",
              ["method", "label", "seed", "worst_objective", "average_objective",
               *OBJECTIVES, "mean_output_length", "policy_reference_kl"], rows)

    # --- mean / std ---------------------------------------------------------
    rows = []
    for method, label in TABLE1_ROWS:
        d = data[method]
        vals = lambda k: [r[k] for r in d["seeds"].values() if r.get(k) is not None]
        w_m, w_s = mean_std(vals("worst_objective"))
        a_m, a_s = mean_std(vals("average_objective"))
        rows.append([method, label, len(d["seeds"]), d["complete"],
                     ";".join(map(str, d["missing_seeds"])) or "",
                     w_m if d["complete"] else "", w_s if d["complete"] else "",
                     a_m if d["complete"] else "", a_s if d["complete"] else ""])
    write_csv(out / "table1_mean_std.csv",
              ["method", "label", "n_seeds", "complete", "missing_seeds",
               "worst_objective_mean", "worst_objective_std",
               "average_objective_mean", "average_objective_std"], rows)

    # --- objective-wise -----------------------------------------------------
    rows = []
    for method, label in TABLE1_ROWS:
        d = data[method]
        for obj in OBJECTIVES:
            v = [r.get("objectives", {}).get(obj) for r in d["seeds"].values()
                 if r.get("objectives", {}).get(obj) is not None]
            m, s = mean_std(v)
            rows.append([method, label, obj, len(v), d["complete"],
                         m if d["complete"] else "", s if d["complete"] else ""])
    write_csv(out / "table2_ultrafeedback_objectives.csv",
              ["method", "label", "objective", "n_seeds", "complete", "mean", "std"], rows)

    # --- solver diagnostics -------------------------------------------------
    rows = []
    for sol in sorted((args.root / "final" / "solve").glob("*/solution.json")):
        s = json.loads(sol.read_text())
        ks = s.get("ks") or {}
        rows.append([sol.parent.name, s.get("representation"), s.get("aggregation"),
                     s.get("min_surplus"), sum(s.get("surplus") or []) or None,
                     s.get("kkt_residual"), s.get("projected_kkt_residual"),
                     s.get("fixed_point_residual"), s.get("extra_map_residual"),
                     s.get("proximal_kl"),
                     s.get("target_log_ratio_identity_residual"),
                     sum(s.get("aggregation_weights_raw") or []) or None,
                     ks.get("rho_star"), ks.get("individual_rationality_violation"),
                     ks.get("normalized_surplus_spread")])
    write_csv(out / "solver_diagnostics.csv",
              ["run", "representation", "aggregation", "min_surplus", "sum_surplus",
               "inverse_surplus_residual", "projected_kkt_residual",
               "fixed_point_residual", "extra_map_residual", "proximal_kl",
               "target_identity_residual", "weight_l1", "ks_rho_star",
               "ks_ir_violation", "ks_normalized_spread"], rows)

    # --- compute budget -----------------------------------------------------
    rows = []
    for rid, rec in sorted(registry.items()):
        rows.append([rid, rec.get("phase"), rec.get("method"), rec.get("seed"),
                     rec.get("status"), rec.get("started_at"), rec.get("ended_at"),
                     len(rec.get("gpus") or []), rec.get("exit_code")])
    write_csv(out / "compute_budget.csv",
              ["run_id", "phase", "method", "seed", "status", "started_at", "ended_at",
               "n_gpus", "exit_code"], rows)

    for name, header in (("table3_matched_components.csv",
                          ["comparison", "metric", "mean_difference", "ci_low", "ci_high",
                           "p_value_holm", "complete"]),
                         ("regression_diagnostics.csv",
                          ["method", "seed", "held_out_normalized_mse", "sign_agreement",
                           "pearson", "spearman", "target_rms", "complete"]),
                         ("cycle_conditioned.csv",
                          ["method", "subset", "worst_objective", "average_objective",
                           "n_prompts", "complete"])):
        if not (out / name).exists():
            write_csv(out / name, header, [])

    # --- TeX ----------------------------------------------------------------
    n_complete = sum(1 for m, _ in TABLE1_ROWS if data[m]["complete"])
    preamble = ("% Generated by scripts/experiments/iclr2027_table1_v2/build_results_tables.py\n"
                "% Do not edit by hand: regenerate from the run registry.\n"
                "% \\pending{} marks a cell whose run has not completed all three seeds.\n"
                "% \\providecommand{\\pending}{\\textit{pending}}\n"
                f"% rows complete: {n_complete}/{len(TABLE1_ROWS)}\n")

    lines = [preamble, "\\begin{tabular}{lcc}", "\\toprule",
             "Method & Worst objective & Average objective \\\\", "\\midrule"]
    for method, label in TABLE1_ROWS:
        d = data[method]
        vals = lambda k: [r[k] for r in d["seeds"].values() if r.get(k) is not None]
        w_m, w_s = mean_std(vals("worst_objective"))
        a_m, a_s = mean_std(vals("average_objective"))
        lines.append(f"{label} & {cell(w_m, w_s, d['complete'])} & "
                     f"{cell(a_m, a_s, d['complete'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "table1_generated.tex").write_text("\n".join(lines) + "\n")

    lines = [preamble, "\\begin{tabular}{l" + "c" * len(OBJECTIVES) + "}", "\\toprule",
             "Method & " + " & ".join(o.replace("_", " ") for o in OBJECTIVES) + " \\\\",
             "\\midrule"]
    for method, label in TABLE1_ROWS:
        d = data[method]
        cells = []
        for obj in OBJECTIVES:
            v = [r.get("objectives", {}).get(obj) for r in d["seeds"].values()
                 if r.get("objectives", {}).get(obj) is not None]
            m, s = mean_std(v)
            cells.append(cell(m, s, d["complete"]))
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "table2_generated.tex").write_text("\n".join(lines) + "\n")

    lines = [preamble, "\\begin{tabular}{lccc}", "\\toprule",
             "Matched comparison & $\\Delta$ worst & 95\\% CI & Holm $p$ \\\\", "\\midrule"]
    for a, b in (("NBPO", "Fixed-reference Nash"), ("NBPO", "BT-RM-Nash"),
                 ("NBPO", "Game-utilitarian"), ("NBPO", "Game-KS"),
                 ("NBPO", "PROSPER / MaxEntBW"),
                 ("BT-RM-Nash", "BT-RM-utilitarian")):
        lines.append(f"{a} vs {b} & \\pending{{}} & \\pending{{}} & \\pending{{}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "table3_generated.tex").write_text("\n".join(lines) + "\n")

    status = {"rows_total": len(TABLE1_ROWS), "rows_complete": n_complete,
              "per_row": {m: {"n_seeds": len(data[m]["seeds"]),
                              "complete": data[m]["complete"],
                              "missing_seeds": data[m]["missing_seeds"]}
                          for m, _ in TABLE1_ROWS},
              "manuscript_update_allowed": n_complete == len(TABLE1_ROWS),
              "note": ("the manuscript table is only updated when every required row has "
                       "three valid seeds; until then these files stand alone and every "
                       "incomplete cell renders as \\pending{}")}
    (out / "table_status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
