import argparse
import csv
from pathlib import Path
from typing import Dict, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: float, digits: int = 4) -> str:
    if value != value:
        return "-"
    return f"{value:.{digits}f}"


def fmt_pct(value: float) -> str:
    if value != value:
        return "-"
    return f"{100.0 * value:.2f}"


def write_markdown(path: Path, rows: List[Dict[str, str]]) -> None:
    headers = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8") as f:
        f.write("# MNPO-style Local RM Benchmark\n\n")
        f.write(
            "No paid API judge is used. Scores are computed with local reward models "
            "on the same prompts and summarized with mean objective score, worst-objective "
            "score, and win rate against the base model.\n\n"
        )
        if not rows:
            f.write("No rows.\n")
            return
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write an MNPO-paper-style local benchmark table.")
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--output_prefix", default=None)
    parser.add_argument("--objectives", default="skywork,athene,armo")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    objectives = [obj.strip() for obj in args.objectives.split(",") if obj.strip()]
    output_prefix = Path(args.output_prefix) if args.output_prefix else result_dir / "mnpo_paper_style"

    per_objective = read_csv(result_dir / "per_objective_scores.csv")
    summary = read_csv(result_dir / "model_summary.csv")

    by_model_obj: Dict[str, Dict[str, Dict[str, str]]] = {}
    for row in per_objective:
        by_model_obj.setdefault(row["model"], {})[row["objective"]] = row

    table_rows: List[Dict[str, str]] = []
    for row in summary:
        model = row["model"]
        out: Dict[str, str] = {
            "Model": model,
            "Prompts": str(int(to_float(row.get("num_prompts", "")))),
        }
        for obj in objectives:
            obj_row = by_model_obj.get(model, {}).get(obj, {})
            out[f"{obj} raw"] = fmt(to_float(obj_row.get("mean_raw_score", "")))
            out[f"{obj} norm"] = fmt(to_float(obj_row.get("mean_prompt_norm_score", "")))
            out[f"{obj} WR vs Base (%)"] = fmt_pct(to_float(obj_row.get("win_rate_vs_baseline", "")))
        out["Avg norm"] = fmt(to_float(row.get("mean_objective_norm_score", "")))
        out["Worst norm"] = fmt(to_float(row.get("min_objective_norm_score", "")))
        out["Std norm"] = fmt(to_float(row.get("std_objective_norm_score", "")))
        out["Avg WR vs Base (%)"] = fmt_pct(to_float(row.get("mean_win_rate_vs_baseline", "")))
        out["Worst WR vs Base (%)"] = fmt_pct(to_float(row.get("min_win_rate_vs_baseline", "")))
        table_rows.append(out)

    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if table_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
            writer.writeheader()
            writer.writerows(table_rows)
    write_markdown(md_path, table_rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
