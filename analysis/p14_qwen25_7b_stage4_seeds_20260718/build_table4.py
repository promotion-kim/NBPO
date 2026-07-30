#!/usr/bin/env python3
"""Build one-seed Qwen2.5-7B Table-3 artifacts from measured JSON."""

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED = {
    "base": "Base",
    "ronpo_os": "RONPO (OS)",
    "inpo_avg": "INPO-avg",
    "sppo_avg": "SPPO-avg",
    "simpo": "SimPO",
    "ipo": "IPO",
    "dpo": "DPO",
    "ht_mnpo_harmless": "HT-MNPO (harml.)",
    "ht_mnpo_helpfulness": "HT-MNPO (help.)",
}
FIELDS = (
    "helpfulness_norm", "harmlessness_norm", "mean_objective_norm_score",
    "mean_prompt_worst_norm_score", "mean_win_rate_vs_base", "min_win_rate_vs_base",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(model: str) -> str:
    return model.removesuffix("_stage4")


def decorate(value: float, field: str, rows: list[dict], latex: bool) -> str:
    ordered = sorted({row[field] for row in rows}, reverse=True)
    text = f"{value:.4f}"
    if value == ordered[0]:
        return f"\\textbf{{{text}}}" if latex else f"**{text}**"
    if len(ordered) > 1 and value == ordered[1]:
        return f"\\underline{{{text}}}" if latex else f"<u>{text}</u>"
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    args = parser.parse_args()
    source = args.result_dir / "ranked_validation_summary.json"
    data = json.loads(source.read_text())
    rows = data["ranking"]
    if data.get("status") != "complete" or data.get("records") != 1000:
        raise RuntimeError("incomplete or wrong-size evaluation")
    by_name = {canonical(row["model"]): row for row in rows}
    if "ronpo_topmass" in by_name:
        raise RuntimeError("RONPO top-mass must not appear in this run")
    missing = set(EXPECTED) - set(by_name)
    rows = [by_name[name] for name in EXPECTED if name in by_name]
    for row in rows:
        if row.get("records") != 1000 or any(not math.isfinite(float(row[field])) for field in FIELDS):
            raise RuntimeError(f"invalid metrics: {row['model']}")
    rows.sort(key=lambda row: row["mean_prompt_worst_norm_score"], reverse=True)

    md = [
        f"# Qwen2.5-7B SafeRLHF Stage-4, seed {args.seed}", "",
        "| Method | Help. | Harmless | Avg | Worst (95% CI) | WR_B | wWR_B |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ci = row["mean_prompt_worst_norm_score_ci95"]
        values = [decorate(row[field], field, rows, False) for field in FIELDS]
        values[3] += f" [{ci[0]:.4f}, {ci[1]:.4f}]"
        md.append(f"| {EXPECTED[canonical(row['model'])]} | " + " | ".join(values) + " |")
    for name in EXPECTED:
        if name in missing:
            md.append(f"| {EXPECTED[name]} | FAILED | FAILED | FAILED | FAILED | FAILED | FAILED |")
    comparisons = data.get("new_arm_paired_comparisons", {})
    paired = comparisons.get("ronpo_os_stage4") or comparisons.get("ronpo_os", {})
    md += [
        "", f"Prompts: {data['records']}. Primary: `{data['primary']}`. {data['normalization']}.",
        "Helpfulness is Beaver reward; harmlessness is negative Beaver cost. Decode seed is 42.",
    ]
    if paired:
        ci = paired["paired_prompt_worst_difference_ci95"]
        md.append(
            f"RONPO-OS minus {paired['worst_comparator']} on paired prompt worst reward: "
            f"{paired['paired_prompt_worst_difference']:.4f} (95% CI [{ci[0]:.4f}, {ci[1]:.4f}])."
        )
    (args.result_dir / f"TABLE3_QWEN25_SEED{args.seed}.md").write_text("\n".join(md) + "\n")

    tex = [
        "\\begin{table}[t]", "\\centering", "\\scriptsize", "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrrrrr}", "\\toprule",
        " & \\multicolumn{2}{c}{Per-objective norm.} & \\multicolumn{4}{c}{Aggregate} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-7}",
        "Method & Help. & Harmless & Avg & Worst & WR$_{\\mathrm{B}}$ & wWR$_{\\mathrm{B}}$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        values = [decorate(row[field], field, rows, True) for field in FIELDS]
        tex.append(f"{EXPECTED[canonical(row['model'])]} & " + " & ".join(values) + " \\\\")
    for name in EXPECTED:
        if name in missing:
            tex.append(EXPECTED[name] + r" & \multicolumn{6}{c}{FAILED (stability gate)} \\")
    tex += [
        "\\bottomrule", "\\end{tabular}%", "}",
        f"\\caption{{Qwen2.5-7B-Instruct on the 1,000-prompt SafeRLHF panel at Stage 4, seed {args.seed}. "
        "Scores use per-prompt min--max normalization across the complete eligible pool; Worst is the mean "
        "of the prompt-level minimum over Beaver helpfulness and harmlessness rewards. Methods marked FAILED "
        "did not reach Stage 4 after a genuine reward-blind stability-gate failure.}",
        f"\\label{{tab:qwen25-saferlhf-stage4-s{args.seed}}}", "\\end{table}",
    ]
    (args.result_dir / f"table3_qwen25_seed{args.seed}.tex").write_text("\n".join(tex) + "\n")
    scope_amendment = args.run_root / "scope_amendment.json"
    provenance = {
        "status": "complete", "metrics_source": str(source), "metrics_sha256": sha256(source),
        "run_lock_sha256": sha256(args.run_root / "run_lock.json"),
        "scope_amendment_sha256": sha256(scope_amendment) if scope_amendment.is_file() else None,
        "scope_amendment_status": "present" if scope_amendment.is_file() else "not present in retry root",
        "records": 1000, "seed": args.seed, "ronpo_topmass_trained": False,
        "stability_failed_or_blocked": sorted(missing),
    }
    (args.result_dir / "TABLE4_PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
