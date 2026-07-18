import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


OBJECTIVE_PAIRS = [
    ("helpfulness", "safety"),
    ("helpfulness", "concise"),
    ("helpfulness", "brevity"),
    ("safety", "concise"),
    ("safety", "brevity"),
]


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def prompt_key(record: Dict[str, Any]) -> str:
    return str(record.get("prompt_id") or record.get("prompt"))


def minmax(values: List[float]) -> List[float]:
    lo = min(values)
    hi = max(values)
    if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) < 1e-12:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def average_ranks(values: List[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        avg_rank = (start + end - 1) / 2.0
        for pos in range(start, end):
            ranks[indexed[pos][0]] = avg_rank
        start = end
    return ranks


def pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman(xs: List[float], ys: List[float]) -> float:
    return pearson(average_ranks(xs), average_ranks(ys))


def percentile(values: List[float], q: float) -> float:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return float("nan")
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    weight = pos - lo
    return clean[lo] * (1.0 - weight) + clean[hi] * weight


def argmax(values: List[float]) -> int:
    return max(range(len(values)), key=lambda idx: values[idx])


def load_objectives(named_paths: List[Tuple[str, Path]]) -> List[Dict[str, Any]]:
    by_objective: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for name, path in named_paths:
        by_objective[name] = {prompt_key(row): row for row in read_jsonl(path)}
    objective_names = [name for name, _ in named_paths]
    keys = set(by_objective[objective_names[0]])
    for name in objective_names[1:]:
        keys &= set(by_objective[name])

    records = []
    for key in sorted(keys):
        first = by_objective[objective_names[0]][key]
        responses = first.get("all_generated_responses")
        if not isinstance(responses, list) or len(responses) < 2:
            continue
        scores: Dict[str, List[float]] = {}
        ok = True
        for name in objective_names:
            row = by_objective[name][key]
            row_scores = row.get("all_rm_scores")
            if (
                row.get("prompt") != first.get("prompt")
                or row.get("all_generated_responses") != responses
                or not isinstance(row_scores, list)
                or len(row_scores) != len(responses)
            ):
                ok = False
                break
            scores[name] = [float(v) for v in row_scores]
        if not ok:
            continue
        records.append(
            {
                "prompt_id": first.get("prompt_id", key),
                "prompt": first.get("prompt", ""),
                "category": first.get("category") or first.get("source") or first.get("split_source") or "unknown",
                "scores": scores,
            }
        )
    return records


def summarize_category(records: List[Dict[str, Any]], category: str) -> Dict[str, Any]:
    rows = [row for row in records if category == "all" or row["category"] == category]
    out: Dict[str, Any] = {"category": category, "prompts": len(rows)}
    if not rows:
        return out

    pair_rhos: Dict[Tuple[str, str], List[float]] = {}
    top1_mismatch = 0
    decoy = 0
    distinct_winners = []
    objective_names = list(rows[0]["scores"])
    for row in rows:
        norm = {name: minmax(values) for name, values in row["scores"].items()}
        winners = {name: argmax(norm[name]) for name in objective_names}
        distinct_winners.append(len(set(winners.values())))
        if "helpfulness" in norm and "safety" in norm:
            if winners["helpfulness"] != winners["safety"]:
                top1_mismatch += 1
            safety_order = sorted(range(len(norm["safety"])), key=lambda idx: norm["safety"][idx])
            bottom_count = max(1, math.ceil(len(safety_order) / 4.0))
            if winners["helpfulness"] in set(safety_order[:bottom_count]):
                decoy += 1
        for left, right in OBJECTIVE_PAIRS:
            if left in norm and right in norm:
                rho = spearman(norm[left], norm[right])
                if math.isfinite(rho):
                    pair_rhos.setdefault((left, right), []).append(rho)

    out["avg_distinct_winners"] = sum(distinct_winners) / len(distinct_winners)
    out["help_safety_top1_mismatch_rate"] = top1_mismatch / len(rows)
    out["help_top_safety_bottom_quartile_decoy_rate"] = decoy / len(rows)
    for (left, right), values in pair_rhos.items():
        prefix = f"spearman_{left}_vs_{right}"
        out[f"{prefix}_mean"] = sum(values) / len(values)
        out[f"{prefix}_median"] = percentile(values, 0.5)
        out[f"{prefix}_p25"] = percentile(values, 0.25)
        out[f"{prefix}_p75"] = percentile(values, 0.75)
    return out


def parse_named_paths(values: List[str]) -> List[Tuple[str, Path]]:
    out = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected name=path, got {value!r}")
        name, path = value.split("=", 1)
        name = name.strip()
        if name == "brevity":
            name = "concise"
        out.append((name, Path(path.strip())))
    return out


def write_outputs(rows: List[Dict[str, Any]], output_dir: Path, title: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with (output_dir / "gate1_conflict_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "gate1_conflict_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    lines = [
        f"# {title}",
        "",
        "| Category | Prompts | Help-Safe rho med | Help-Concise rho med | Decoy rate | Top-1 mismatch | Avg distinct winners |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {category} | {prompts} | {hs:.3f} | {hc:.3f} | {decoy:.3f} | {mismatch:.3f} | {distinct:.3f} |".format(
                category=row.get("category", ""),
                prompts=row.get("prompts", 0),
                hs=row.get("spearman_helpfulness_vs_safety_median", float("nan")),
                hc=row.get("spearman_helpfulness_vs_concise_median", float("nan")),
                decoy=row.get("help_top_safety_bottom_quartile_decoy_rate", float("nan")),
                mismatch=row.get("help_safety_top1_mismatch_rate", float("nan")),
                distinct=row.get("avg_distinct_winners", float("nan")),
            )
        )
    lines.extend(
        [
            "",
            "Preset Gate 1 thresholds for red-team subsets: median rho(help,safe) <= 0.3, decoy rate >= 0.20, top-1 mismatch >= 0.40. For global helpful-concise, median rho < 0 indicates the expected helpfulness/length tension.",
        ]
    )
    (output_dir / "gate1_conflict_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 1 conflict diagnostics for RONPO multi-objective pools.")
    parser.add_argument("--scored_files", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--title", default="RONPO Gate 1 Conflict Diagnostics")
    args = parser.parse_args()

    records = load_objectives(parse_named_paths(args.scored_files))
    categories = ["all"] + sorted({row["category"] for row in records})
    rows = [summarize_category(records, category) for category in categories]
    write_outputs(rows, Path(args.output_dir), args.title)
    print(f"[done] analyzed {len(records)} prompts -> {args.output_dir}")


if __name__ == "__main__":
    main()
