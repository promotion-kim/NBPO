import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


WORD_RE = re.compile(r"\b[\w'-]+\b")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prompt_key(record: Dict[str, Any]) -> str:
    return str(record.get("prompt_id") or record.get("prompt"))


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def banded_brevity_score(text: str, target_words: int, tolerance_words: int) -> float:
    """Length reward with a plateau around target to avoid the trivial shortest-answer optimum."""
    length = word_count(text)
    lower = max(1, target_words - tolerance_words)
    upper = target_words + tolerance_words
    if lower <= length <= upper:
        return 1.0
    if length < lower:
        distance = lower - length
    else:
        distance = length - upper
    return max(0.0, 1.0 - distance / max(float(target_words), 1.0))


def minmax(values: List[float]) -> List[float]:
    lo = min(values)
    hi = max(values)
    if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) < 1e-12:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def argmax(values: List[float]) -> int:
    return max(range(len(values)), key=lambda idx: values[idx])


def argmin(values: List[float]) -> int:
    return min(range(len(values)), key=lambda idx: values[idx])


def objective_row(base: Dict[str, Any], scores: List[float]) -> Dict[str, Any]:
    responses = base["all_generated_responses"]
    best = argmax(scores)
    worst = argmin(scores)
    return {
        "prompt": base["prompt"],
        "prompt_id": base.get("prompt_id"),
        "all_generated_responses": responses,
        "all_rm_scores": [float(score) for score in scores],
        "chosen": [
            {"role": "user", "content": base["prompt"]},
            {"role": "assistant", "content": responses[best]},
        ],
        "rejected": [
            {"role": "user", "content": base["prompt"]},
            {"role": "assistant", "content": responses[worst]},
        ],
    }


def process_split(
    split: str,
    helpfulness_path: Path,
    safety_path: Path,
    output_dir: Path,
    target_words: int,
    tolerance_words: int,
) -> Dict[str, Any]:
    helpful_by_key = {prompt_key(row): row for row in read_jsonl(helpfulness_path)}
    safety_by_key = {prompt_key(row): row for row in read_jsonl(safety_path)}
    common = sorted(set(helpful_by_key) & set(safety_by_key))

    helpful_rows = []
    safety_rows = []
    brevity_rows = []
    norm_points = {"helpfulness": [], "safety": [], "brevity": []}
    distinct_winner_counts = []
    skipped = 0
    response_count = 0

    for key in common:
        helpful = helpful_by_key[key]
        safety = safety_by_key[key]
        responses = helpful.get("all_generated_responses")
        helpful_scores = helpful.get("all_rm_scores")
        safety_scores = safety.get("all_rm_scores")
        if (
            not isinstance(responses, list)
            or not isinstance(helpful_scores, list)
            or not isinstance(safety_scores, list)
            or len(responses) < 2
            or len(responses) != len(helpful_scores)
            or len(responses) != len(safety_scores)
            or helpful.get("prompt") != safety.get("prompt")
            or responses != safety.get("all_generated_responses")
        ):
            skipped += 1
            continue

        helpful_values = [float(v) for v in helpful_scores]
        safety_values = [float(v) for v in safety_scores]
        brevity_values = [
            banded_brevity_score(str(response), target_words, tolerance_words)
            for response in responses
        ]

        helpful_rows.append(objective_row(helpful, helpful_values))
        safety_rows.append(objective_row(helpful, safety_values))
        brevity_rows.append(objective_row(helpful, brevity_values))
        response_count += len(responses)

        normalized = {
            "helpfulness": minmax(helpful_values),
            "safety": minmax(safety_values),
            "brevity": minmax(brevity_values),
        }
        for name, values in normalized.items():
            norm_points[name].extend(values)
        distinct_winner_counts.append(len({argmax(values) for values in normalized.values()}))

    scored_dir = output_dir / "scored"
    write_jsonl(scored_dir / f"{split}_helpfulness.jsonl", helpful_rows)
    write_jsonl(scored_dir / f"{split}_safety.jsonl", safety_rows)
    write_jsonl(scored_dir / f"{split}_brevity.jsonl", brevity_rows)

    pairs = [("helpfulness", "safety"), ("helpfulness", "brevity"), ("safety", "brevity")]
    correlations = {
        f"{left}_vs_{right}": pearson(norm_points[left], norm_points[right])
        for left, right in pairs
    }
    avg_distinct = (
        sum(distinct_winner_counts) / len(distinct_winner_counts)
        if distinct_winner_counts
        else float("nan")
    )
    return {
        "split": split,
        "prompts": len(helpful_rows),
        "responses": response_count,
        "skipped": skipped,
        "target_words": target_words,
        "tolerance_words": tolerance_words,
        "avg_distinct_objective_winners": avg_distinct,
        **correlations,
    }


def write_summary(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "conflict_objective_summary.csv"
    json_path = output_dir / "conflict_objective_summary.json"
    keys = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    report = output_dir / "report.md"
    lines = [
        "# RONPO Safety Conflict Objective Gate",
        "",
        "Objectives: helpfulness uses the existing reward-model score, safety uses the configured safety reward head, and brevity uses a target-length band rather than a monotone shortest-answer reward.",
        "",
        "| Split | Prompts | Responses | Avg distinct winners | Help-Safety r | Help-Brevity r | Safety-Brevity r |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {split} | {prompts} | {responses} | {avg:.3f} | {hs:.3f} | {hb:.3f} | {sb:.3f} |".format(
                split=row["split"],
                prompts=row["prompts"],
                responses=row["responses"],
                avg=row["avg_distinct_objective_winners"],
                hs=row["helpfulness_vs_safety"],
                hb=row["helpfulness_vs_brevity"],
                sb=row["safety_vs_brevity"],
            )
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build non-heuristic conflict objectives from helpfulness, guard safety, and banded brevity."
    )
    parser.add_argument("--train_helpfulness", required=True)
    parser.add_argument("--test_helpfulness", required=True)
    parser.add_argument("--train_safety", required=True)
    parser.add_argument("--test_safety", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_words", type=int, default=180)
    parser.add_argument("--tolerance_words", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = [
        process_split(
            "train",
            Path(args.train_helpfulness),
            Path(args.train_safety),
            output_dir,
            args.target_words,
            args.tolerance_words,
        ),
        process_split(
            "test",
            Path(args.test_helpfulness),
            Path(args.test_safety),
            output_dir,
            args.target_words,
            args.tolerance_words,
        ),
    ]
    write_summary(output_dir, rows)
    print(f"[done] wrote objective files and report under {output_dir}")


if __name__ == "__main__":
    main()
