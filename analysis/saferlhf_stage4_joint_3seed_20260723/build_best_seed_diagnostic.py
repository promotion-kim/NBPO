#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


LABELS = {
    "base": "Base",
    "ronpo_os": "RONPO (OS)",
    "ronpo_topmass": "RONPO (top-mass)",
    "inpo_avg": "INPO (avg)",
    "sppo_avg": "SPPO (avg)",
    "simpo": "SimPO",
    "ipo": "IPO",
    "dpo": "DPO",
    "ht_mnpo_harmless": "HT-MNPO (harmless)",
    "ht_mnpo_helpfulness": "HT-MNPO (helpful)",
}


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def fmt(value):
    return f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    paired = load_json(args.result_dir / "paired_summary.json")
    independent = load_json(args.result_dir / "independent_summary.json")
    with (args.result_dir / "per_policy_scores.csv").open() as handle:
        policy_rows = {row["policy"]: row for row in csv.DictReader(handle)}

    ronpo = paired["method_summary"]["ronpo_os"]["hard_pairwise_floor"]["per_seed"]
    best_seed = max(ronpo, key=lambda seed: ronpo[seed]["mean"])
    methods = [key for key in LABELS if key not in {"base", "ronpo_os"}]
    baselines = [key for key in methods if not key.startswith("ronpo_")]
    seed_scan = []
    for seed in sorted(ronpo):
        best_baseline = max(
            baselines,
            key=lambda method: paired["method_summary"][method]["hard_pairwise_floor"]
            ["per_seed"][seed]["mean"],
        )
        contrast = paired["contrasts"][f"ronpo_os_minus_{best_baseline}"][
            "hard_pairwise_floor"
        ]["per_seed"][seed]
        seed_scan.append(
            {
                "seed": int(seed),
                "ronpo_hard_pairwise_floor": ronpo[seed]["mean"],
                "ronpo_ci95": ronpo[seed]["ci95"],
                "best_non_ronpo": best_baseline,
                "best_non_ronpo_hard_pairwise_floor": paired["method_summary"][
                    best_baseline
                ]["hard_pairwise_floor"]["per_seed"][seed]["mean"],
                "ronpo_delta": contrast["delta"],
                "ronpo_delta_ci95": contrast["ci95"],
            }
        )

    table = []
    for method in ["ronpo_os", *methods, "base"]:
        row = policy_rows["base" if method == "base" else f"{method}_s{best_seed}"]
        table.append(
            {
                "method": method,
                "label": LABELS[method],
                "helpfulness_raw": float(row["beaver_helpfulness_raw"]),
                "harmlessness_raw": float(row["beaver_harmlessness_raw"]),
                "legacy_average_norm": float(row["legacy_avg_norm"]),
                "legacy_worst_norm": float(row["legacy_worst_norm"]),
                "hard_pairwise_floor": float(row["hard_pairwise_floor"]),
                "soft_pairwise_floor": float(row["soft_pairwise_floor"]),
                "independent_worst_win_rate": (
                    0.5
                    if method == "base"
                    else independent["method_summary"][method]["per_seed"][best_seed][
                        "worst_marginal_win_rate"
                    ]
                ),
            }
        )
    table.sort(key=lambda row: row["hard_pairwise_floor"], reverse=True)
    output = {
        "status": "post_hoc_diagnostic_not_confirmatory",
        "selection_rule": "maximize RONPO-OS hard pairwise floor over existing seeds 42, 43, and 44",
        "selected_seed": int(best_seed),
        "seed_scan": seed_scan,
        "table": table,
        "warning": (
            "The selected-seed result is a winner's-curse-inflated diagnostic upper "
            "bound and must not replace the preregistered three-seed result."
        ),
    }
    (args.result_dir / "best_seed_diagnostic.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )

    lines = [
        "# Post-hoc best-seed diagnostic",
        "",
        "**Not confirmatory.** Seed 42 was selected after observing the existing",
        "seed 42/43/44 hard pairwise floors. This is a diagnostic upper bound and",
        "must not replace the preregistered three-seed result.",
        "",
        "## Seed selection",
        "",
        "| Seed | RONPO floor | 95% CI | Best non-RONPO | Baseline floor | Delta | Delta 95% CI |",
        "|---:|---:|:---:|:---|---:|---:|:---:|",
    ]
    for row in seed_scan:
        lines.append(
            f"| {row['seed']} | {fmt(row['ronpo_hard_pairwise_floor'])} | "
            f"[{fmt(row['ronpo_ci95'][0])}, {fmt(row['ronpo_ci95'][1])}] | "
            f"{LABELS[row['best_non_ronpo']]} | "
            f"{fmt(row['best_non_ronpo_hard_pairwise_floor'])} | "
            f"{row['ronpo_delta']:+.4f} | "
            f"[{row['ronpo_delta_ci95'][0]:+.4f}, "
            f"{row['ronpo_delta_ci95'][1]:+.4f}] |"
        )
    lines.extend(
        [
            "",
            f"## Selected seed {best_seed} full comparison",
            "",
            "| Method | Help. raw | Harmless raw | Avg norm. | Legacy Worst | Pair floor | Soft floor | Independent wWR_B |",
            "|:---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    columns = [
        "helpfulness_raw",
        "harmlessness_raw",
        "legacy_average_norm",
        "legacy_worst_norm",
        "hard_pairwise_floor",
        "soft_pairwise_floor",
        "independent_worst_win_rate",
    ]
    maxima = {column: max(row[column] for row in table) for column in columns}
    for row in table:
        values = [
            (
                f"**{fmt(row[column])}**"
                if row[column] == maxima[column]
                else fmt(row[column])
            )
            for column in columns
        ]
        lines.append(f"| {row['label']} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Selection used only the hard pairwise floor. Bold marks the best value",
            "within the selected-seed comparison for each column.",
            "The frozen evaluation has 1,000 prompts and uses no new training or decoding.",
            "",
        ]
    )
    (args.result_dir / "BEST_SEED_DIAGNOSTIC.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
