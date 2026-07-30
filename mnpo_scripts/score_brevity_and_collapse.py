import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List


WORD_RE = re.compile(r"\b[\w'-]+\b")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def read_records(path: Path) -> Iterable[Dict[str, Any]]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as f:
        if suffix in {".jsonl", ".jsonlines"}:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        elif suffix == ".json":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON list in {path}")
            yield from data
        else:
            raise ValueError(f"Unsupported suffix for {path}")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def banded_brevity_score(text: str, target_words: int, tolerance_words: int) -> float:
    length = word_count(text)
    lower = max(1, target_words - tolerance_words)
    upper = target_words + tolerance_words
    if lower <= length <= upper:
        return 1.0
    distance = lower - length if length < lower else length - upper
    return max(0.0, 1.0 - distance / max(float(target_words), 1.0))


def max_token_run(text: str) -> int:
    tokens = TOKEN_RE.findall(text.lower())
    best = 0
    prev = None
    cur = 0
    for token in tokens:
        if token == prev:
            cur += 1
        else:
            prev = token
            cur = 1
        best = max(best, cur)
    return best


def unique_token_ratio(text: str) -> float:
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        return 0.0
    return len(set(tokens)) / float(len(tokens))


def objective_row(record: Dict[str, Any], scores: List[float]) -> Dict[str, Any]:
    responses = record["all_generated_responses"]
    best = max(range(len(scores)), key=lambda i: scores[i])
    worst = min(range(len(scores)), key=lambda i: scores[i])
    return {
        **record,
        "all_rm_scores": [float(score) for score in scores],
        "chosen": [
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": responses[best]},
        ],
        "rejected": [
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": responses[worst]},
        ],
    }


def pct(values: List[bool]) -> float:
    return sum(1.0 for value in values if value) / len(values) if values else float("nan")


def write_collapse_summary(path: Path, model_names: List[str], records: List[Dict[str, Any]]) -> None:
    rows = []
    for idx, model in enumerate(model_names):
        texts = [str(record["all_generated_responses"][idx]) for record in records]
        words = [word_count(text) for text in texts]
        chars = [len(text) for text in texts]
        runs = [max_token_run(text) for text in texts]
        uniq = [unique_token_ratio(text) for text in texts]
        rows.append(
            {
                "model": model,
                "num_prompts": len(texts),
                "mean_words": mean(words) if words else float("nan"),
                "median_words": median(words) if words else float("nan"),
                "mean_chars": mean(chars) if chars else float("nan"),
                "empty_rate": pct([w == 0 for w in words]),
                "very_short_rate_lt10_words": pct([w < 10 for w in words]),
                "long_rate_gt512_words": pct([w > 512 for w in words]),
                "mean_max_token_run": mean(runs) if runs else float("nan"),
                "max_token_run_ge10_rate": pct([run >= 10 for run in runs]),
                "max_token_run_ge20_rate": pct([run >= 20 for run in runs]),
                "mean_unique_token_ratio": mean(uniq) if uniq else float("nan"),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "num_prompts",
        "mean_words",
        "median_words",
        "mean_chars",
        "empty_rate",
        "very_short_rate_lt10_words",
        "long_rate_gt512_words",
        "mean_max_token_run",
        "max_token_run_ge10_rate",
        "max_token_run_ge20_rate",
        "mean_unique_token_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score generated model responses with the training brevity objective.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--collapse_csv", required=True)
    parser.add_argument("--target_words", type=int, default=180)
    parser.add_argument("--tolerance_words", type=int, default=80)
    args = parser.parse_args()

    records = list(read_records(Path(args.input_file)))
    out_rows = []
    for record in records:
        responses = [str(response) for response in record.get("all_generated_responses", [])]
        if not responses:
            continue
        scores = [
            banded_brevity_score(response, args.target_words, args.tolerance_words)
            for response in responses
        ]
        out_rows.append(objective_row(record, scores))

    write_jsonl(Path(args.output_file), out_rows)
    if out_rows:
        model_names = list(out_rows[0].get("response_model_names", []))
        if model_names:
            write_collapse_summary(Path(args.collapse_csv), model_names, out_rows)
    print(f"[done] wrote brevity scores to {args.output_file}")
    print(f"[done] wrote collapse diagnostics to {args.collapse_csv}")


if __name__ == "__main__":
    main()
