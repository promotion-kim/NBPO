import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


WORD_RE = re.compile(r"\b[\w'-]+\b")


def read_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() in {".jsonl", ".jsonlines"}:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        elif path.suffix.lower() == ".json":
            data = json.load(f)
            if isinstance(data, list):
                yield from data
            elif isinstance(data, dict):
                for value in data.values():
                    if isinstance(value, list):
                        yield from value
                        return
                yield data
            else:
                raise ValueError(f"Unsupported JSON root in {path}")
        else:
            raise ValueError(f"Unsupported file suffix: {path}")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def sentence_count(text: str) -> int:
    chunks = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    return len(chunks)


def json_like(text: str) -> bool:
    stripped = text.strip()
    candidates = [stripped]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(chunk.strip() for chunk in fenced)
    for candidate in candidates:
        if not candidate or candidate[0] not in "[{":
            continue
        try:
            json.loads(candidate)
            return True
        except Exception:
            continue
    return False


def bullet_or_numbered_lines(text: str) -> int:
    count = 0
    for line in text.splitlines():
        line = line.strip()
        if re.match(r"^([-*+]|\d+[.)])\s+", line):
            count += 1
    return count


def detect_format_rules(prompt: str) -> List[Tuple[str, Optional[int]]]:
    p = prompt.lower()
    rules: List[Tuple[str, Optional[int]]] = []
    if "json" in p:
        rules.append(("json", None))
    if any(token in p for token in ["bullet", "bulleted", "numbered list", "list of", "enumerate"]):
        rules.append(("list", None))
    if any(token in p for token in ["yes or no", "yes/no", "answer yes", "answer no"]):
        rules.append(("yes_no", None))
    elif re.search(r"\b(is|are|do|does|did|can|could|should|would|will|was|were)\b.+\?", p):
        rules.append(("yes_no", None))
    if any(token in p for token in ["one sentence", "single sentence"]):
        rules.append(("max_sentences", 1))
    if any(token in p for token in ["two sentences", "2 sentences"]):
        rules.append(("max_sentences", 2))
    m = re.search(r"(?:no more than|at most|under)\s+(\d+)\s+words", p)
    if m:
        rules.append(("max_words", int(m.group(1))))
    if any(token in p for token in ["one word", "single word"]):
        rules.append(("max_words", 3))
    return rules


def format_score(prompt: str, response: str) -> Tuple[float, List[str]]:
    rules = detect_format_rules(prompt)
    if not rules:
        return 0.5, []

    passed = 0.0
    names: List[str] = []
    stripped = response.strip()
    for name, value in rules:
        names.append(name if value is None else f"{name}_{value}")
        if name == "json":
            passed += 1.0 if json_like(response) else 0.0
        elif name == "list":
            passed += 1.0 if bullet_or_numbered_lines(response) >= 2 else 0.0
        elif name == "yes_no":
            first = stripped.split(None, 1)[0].lower().strip(",:;.") if stripped else ""
            passed += 1.0 if first in {"yes", "no"} else 0.0
        elif name == "max_sentences":
            passed += 1.0 if sentence_count(response) <= int(value or 1) else 0.0
        elif name == "max_words":
            passed += 1.0 if word_count(response) <= int(value or 1) else 0.0
    return passed / len(rules), names


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
    return max(range(len(values)), key=lambda i: values[i])


def row_for_objective(base: Dict[str, Any], scores: List[float]) -> Dict[str, Any]:
    responses = base["all_generated_responses"]
    best = argmax(scores)
    worst = min(range(len(scores)), key=lambda i: scores[i])
    return {
        "prompt": base["prompt"],
        "prompt_id": base.get("prompt_id"),
        "all_generated_responses": responses,
        "all_rm_scores": [float(s) for s in scores],
        "chosen": [
            {"role": "user", "content": base["prompt"]},
            {"role": "assistant", "content": responses[best]},
        ],
        "rejected": [
            {"role": "user", "content": base["prompt"]},
            {"role": "assistant", "content": responses[worst]},
        ],
    }


def process_split(split: str, input_path: Path, output_dir: Path) -> Dict[str, Any]:
    records = list(read_records(input_path))
    helpful_rows = []
    brevity_rows = []
    format_rows = []

    norm_points: Dict[str, List[float]] = {"helpfulness": [], "brevity": [], "format": []}
    distinct_winner_counts: List[int] = []
    format_active = 0
    prompt_count = 0
    response_count = 0

    for record in records:
        responses = record.get("all_generated_responses")
        helpful = record.get("all_rm_scores")
        if not isinstance(responses, list) or not isinstance(helpful, list) or len(responses) != len(helpful):
            continue
        prompt = str(record.get("prompt", ""))
        helpful_scores = [float(v) for v in helpful]
        brevity_scores = [-float(word_count(str(resp))) for resp in responses]
        format_scores = []
        active_any = False
        for resp in responses:
            score, rules = format_score(prompt, str(resp))
            format_scores.append(score)
            active_any = active_any or bool(rules)
        if active_any:
            format_active += 1

        helpful_rows.append(row_for_objective(record, helpful_scores))
        brevity_rows.append(row_for_objective(record, brevity_scores))
        format_rows.append(row_for_objective(record, format_scores))

        prompt_count += 1
        response_count += len(responses)
        normalized = {
            "helpfulness": minmax(helpful_scores),
            "brevity": minmax(brevity_scores),
            "format": minmax(format_scores),
        }
        for name, vals in normalized.items():
            norm_points[name].extend(vals)
        winners = {argmax(vals) for vals in normalized.values()}
        distinct_winner_counts.append(len(winners))

    split_dir = output_dir / "scored"
    write_jsonl(split_dir / f"{split}_helpfulness.jsonl", helpful_rows)
    write_jsonl(split_dir / f"{split}_brevity.jsonl", brevity_rows)
    write_jsonl(split_dir / f"{split}_format.jsonl", format_rows)

    pairs = [("helpfulness", "brevity"), ("helpfulness", "format"), ("brevity", "format")]
    correlations = {
        f"{a}_vs_{b}": pearson(norm_points[a], norm_points[b])
        for a, b in pairs
    }
    return {
        "split": split,
        "input": str(input_path),
        "prompts": prompt_count,
        "responses": response_count,
        "format_active_prompts": format_active,
        "format_active_prompt_rate": format_active / prompt_count if prompt_count else 0.0,
        "mean_distinct_objective_winners": sum(distinct_winner_counts) / len(distinct_winner_counts)
        if distinct_winner_counts
        else 0.0,
        "prompts_with_objective_winner_disagreement": sum(1 for v in distinct_winner_counts if v > 1),
        "objective_winner_disagreement_rate": sum(1 for v in distinct_winner_counts if v > 1) / len(distinct_winner_counts)
        if distinct_winner_counts
        else 0.0,
        **correlations,
    }


def write_summary(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "conflict_objective_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    keys = sorted({k for row in rows for k in row})
    with (output_dir / "conflict_objective_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Conflict Objective Pool Diagnostic",
        "",
        "Objectives: helpfulness uses the existing local reward-model score, brevity uses negative word count, and format uses deterministic prompt-conditioned rule checks. All correlations below are computed after prompt-wise min-max normalization across the candidate responses.",
        "",
        "| Split | Prompts | Responses | Format-active prompts | Winner disagreement | Distinct winners | Helpfulness-Brevity r | Helpfulness-Format r | Brevity-Format r |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {split} | {prompts} | {responses} | {fmt:.2%} | {dis:.2%} | {dw:.2f} | {hb:.4f} | {hf:.4f} | {bf:.4f} |".format(
                split=row["split"],
                prompts=row["prompts"],
                responses=row["responses"],
                fmt=row["format_active_prompt_rate"],
                dis=row["objective_winner_disagreement_rate"],
                dw=row["mean_distinct_objective_winners"],
                hb=row["helpfulness_vs_brevity"],
                hf=row["helpfulness_vs_format"],
                bf=row["brevity_vs_format"],
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: low or negative correlations and frequent objective-winner disagreement indicate that the candidate pool contains genuine objective conflict. This is the precondition for testing whether RONPO raises the worst-objective floor rather than only optimizing an average reward.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build conflict-objective scored files from an existing scored response pool.")
    parser.add_argument("--train_helpfulness", required=True)
    parser.add_argument("--test_helpfulness", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = [
        process_split("train", Path(args.train_helpfulness), output_dir),
        process_split("test", Path(args.test_helpfulness), output_dir),
    ]
    write_summary(output_dir, rows)
    print(f"Wrote conflict objective files and report to {output_dir}")


if __name__ == "__main__":
    main()
