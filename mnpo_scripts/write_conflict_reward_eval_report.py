import argparse
import csv
from pathlib import Path
from typing import Dict, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fnum(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def fmt(value: str, digits: int = 3) -> str:
    return f"{fnum(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a compact markdown report for conflict reward evaluation.")
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--collapse_csv", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--title", default="RONPO Conflict Reward Evaluation")
    parser.add_argument("--generation_config", default="")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    summary = read_csv(result_dir / "model_summary.csv")
    objectives = read_csv(result_dir / "per_objective_scores.csv")
    collapse = {row["model"]: row for row in read_csv(Path(args.collapse_csv))}

    summary.sort(
        key=lambda row: fnum(row.get("mean_primary_prompt_worst_norm_score", "nan")),
        reverse=True,
    )
    objective_by_model: Dict[str, Dict[str, Dict[str, str]]] = {}
    for row in objectives:
        objective_by_model.setdefault(row["model"], {})[row["objective"]] = row

    objective_names = sorted({row["objective"] for row in objectives})
    lines = [
        f"# {args.title}",
        "",
        f"- Result dir: `{result_dir}`",
    ]
    if args.generation_config:
        lines.append(f"- Generation config: {args.generation_config}")
    if summary:
        lines.append(f"- Prompts: {summary[0].get('num_prompts', 'unknown')}")
    lines.extend(
        [
            "",
            "## Model Summary",
            "",
            "| Rank | Model | Help--brevity worst [95% CI] | Help--brevity avg | All-three worst | Min objective mean | Mean win vs base | Max-run>=20 | Mean words |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(summary, start=1):
        model = row["model"]
        collapse_row = collapse.get(model, {})
        mean_win = row.get("mean_win_rate_vs_baseline", "")
        lines.append(
            "| {rank} | `{model}` | {worst} | {avg} | {allworst} | {minobj} | {win} | {run20} | {words} |".format(
                rank=rank,
                model=model,
                worst="{point} [{low}, {high}]".format(
                    point=fmt(row.get("mean_primary_prompt_worst_norm_score", "nan")),
                    low=fmt(row.get("mean_primary_prompt_worst_norm_score_ci95_low", "nan")),
                    high=fmt(row.get("mean_primary_prompt_worst_norm_score_ci95_high", "nan")),
                ),
                avg=fmt(row.get("mean_primary_prompt_avg_norm_score", "nan")),
                allworst=fmt(row.get("mean_prompt_worst_norm_score", "nan")),
                minobj=fmt(row.get("min_objective_norm_score", "nan")),
                win="-" if mean_win == "" else fmt(mean_win),
                run20=fmt(collapse_row.get("max_token_run_ge20_rate", "nan")),
                words=fmt(collapse_row.get("mean_words", "nan"), 1),
            )
        )

    lines.extend(["", "## Per Objective", ""])
    header = "| Model | " + " | ".join(f"{name} norm" for name in objective_names) + " |"
    divider = "|---|" + "|".join("---:" for _ in objective_names) + "|"
    lines.extend([header, divider])
    for row in summary:
        model = row["model"]
        cells = [model]
        for objective in objective_names:
            obj_row = objective_by_model.get(model, {}).get(objective, {})
            cells.append(fmt(obj_row.get("mean_prompt_norm_score", "nan")))
        lines.append("| " + " | ".join(f"`{cell}`" if idx == 0 else cell for idx, cell in enumerate(cells)) + " |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `Help--brevity worst` is the pre-registered primary metric: the mean over prompts of the minimum prompt-wise normalized helpfulness and brevity score.",
            "- `All-three worst` includes helpfulness, the reported safety scorer, and brevity and is always reported as a secondary robustness view.",
            "- `Min objective mean` is the minimum across objective-level mean normalized scores.",
            "- Collapse diagnostics are descriptive only; high `Max-run>=20` indicates repeated-token runs and should be inspected manually.",
        ]
    )

    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
