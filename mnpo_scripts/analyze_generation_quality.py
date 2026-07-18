import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


WORD_RE = re.compile(r"\S+")


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of merged generation records: {path}")
    return data


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def median(values: List[float]) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def distinct_ngram_ratio(words: List[str], n: int) -> float:
    if len(words) < n:
        return 1.0 if words else 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def max_line_repeat_fraction(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    counts = Counter(lines)
    return max(counts.values()) / len(lines)


def model_records(records: Iterable[Dict[str, Any]]) -> Dict[str, List[str]]:
    by_model: Dict[str, List[str]] = {}
    for record in records:
        names = record.get("response_model_names")
        responses = record.get("all_generated_responses")
        if not isinstance(names, list) or not isinstance(responses, list):
            continue
        if len(names) != len(responses):
            raise ValueError("response_model_names and all_generated_responses length mismatch")
        for name, response in zip(names, responses):
            by_model.setdefault(str(name), []).append("" if response is None else str(response))
    return by_model


def summarize(name: str, responses: List[str], long_char_threshold: int) -> Dict[str, Any]:
    char_lengths = [len(r) for r in responses]
    word_lists = [WORD_RE.findall(r) for r in responses]
    word_lengths = [len(w) for w in word_lists]
    distinct_4 = [distinct_ngram_ratio([x.lower() for x in words], 4) for words in word_lists]
    line_repeat = [max_line_repeat_fraction(r) for r in responses]
    empty = [1.0 if not r.strip() else 0.0 for r in responses]
    long = [1.0 if len(r) >= long_char_threshold else 0.0 for r in responses]
    suspect_repeat = [1.0 if (d4 < 0.35 and wl >= 80) or lr >= 0.35 else 0.0 for d4, wl, lr in zip(distinct_4, word_lengths, line_repeat)]
    return {
        "model": name,
        "n": len(responses),
        "mean_chars": mean(char_lengths),
        "median_chars": median(char_lengths),
        "p90_chars": percentile(char_lengths, 0.90),
        "max_chars": max(char_lengths) if char_lengths else 0,
        "mean_words": mean(word_lengths),
        "median_words": median(word_lengths),
        "empty_rate": mean(empty),
        "long_rate": mean(long),
        "mean_distinct_4gram_ratio": mean(distinct_4),
        "median_distinct_4gram_ratio": median(distinct_4),
        "mean_max_line_repeat_fraction": mean(line_repeat),
        "suspect_repetition_rate": mean(suspect_repeat),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize generation length and simple repetition diagnostics.")
    parser.add_argument("--merged_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--long_char_threshold", type=int, default=12000)
    args = parser.parse_args()

    records = load_json(Path(args.merged_file))
    rows = [
        summarize(name, responses, args.long_char_threshold)
        for name, responses in model_records(records).items()
    ]
    outdir = Path(args.output_dir)
    write_csv(outdir / "generation_quality_summary.csv", rows)
    with (outdir / "generation_quality_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Wrote generation quality summary for {len(rows)} models to {outdir}")


if __name__ == "__main__":
    main()
