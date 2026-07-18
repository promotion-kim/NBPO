import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def read_records(path: str) -> Iterable[Dict[str, Any]]:
    suffix = Path(path).suffix.lower()
    with open(path, "r", encoding="utf-8") as f:
        if suffix in {".jsonl", ".jsonlines"}:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        elif suffix == ".json":
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
                raise ValueError(f"Unsupported JSON structure in {path}")
        else:
            raise ValueError(f"Unsupported file suffix for {path}")


def parse_named_path(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"Expected objective=path, got {value!r}")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"Expected non-empty objective=path, got {value!r}")
    return name, path


def make_messages(prompt: str, response: str) -> List[Dict[str, str]]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def minmax(values: List[float]) -> List[float]:
    lo = min(values)
    hi = max(values)
    if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) < 1e-12:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def rank_normalize(values: List[float]) -> List[float]:
    if len(values) == 1:
        return [0.5]
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for rank, idx in enumerate(order):
        ranks[idx] = rank / float(len(values) - 1)
    return ranks


def normalize(values: List[float], mode: str) -> List[float]:
    values = [float(v) for v in values]
    if mode == "none":
        return values
    if mode == "minmax":
        return minmax(values)
    if mode == "rank":
        return rank_normalize(values)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def argmax(values: List[float]) -> int:
    return max(range(len(values)), key=lambda i: values[i])


def argmin(values: List[float]) -> int:
    return min(range(len(values)), key=lambda i: values[i])


def build_rows(
    objective_name: str,
    input_path: str,
    normalization: str,
    target_mode: str,
    tie_threshold: float,
    max_rows: int = 0,
) -> List[Dict[str, Any]]:
    rows = []
    for record in read_records(input_path):
        prompt = record["prompt"]
        responses = record.get("all_generated_responses")
        scores = record.get("all_rm_scores")
        if not isinstance(responses, list) or not isinstance(scores, list):
            continue
        if len(responses) < 2 or len(responses) != len(scores):
            continue

        raw_scores = [float(s) for s in scores]
        norm_scores = normalize(raw_scores, normalization)
        chosen_idx = argmax(norm_scores)
        rejected_idx = argmin(norm_scores)
        if chosen_idx == rejected_idx:
            continue

        normalized_gap = float(norm_scores[chosen_idx] - norm_scores[rejected_idx])
        raw_gap = float(raw_scores[chosen_idx] - raw_scores[rejected_idx])
        if abs(raw_gap) <= tie_threshold:
            continue

        if target_mode == "reward_gap":
            target = raw_gap
        elif target_mode in {"score_gap", "normalized_gap"}:
            target = normalized_gap
        elif target_mode == "unit":
            target = 1.0
        else:
            raise ValueError(f"Unsupported target_mode: {target_mode}")

        out = dict(record)
        out["chosen"] = make_messages(prompt, responses[chosen_idx])
        out["rejected"] = make_messages(prompt, responses[rejected_idx])
        out["chosen_index"] = int(chosen_idx)
        out["rejected_index"] = int(rejected_idx)
        out["pair_source"] = "ht_mnpo_player_objective"
        out["ht_target"] = float(target)
        out["ht_objective_name"] = objective_name
        out["ht_objective_gap"] = float(raw_gap)
        out["ht_normalized_gap"] = float(normalized_gap)
        out["ht_target_mode"] = target_mode
        out["ht_score_normalization"] = normalization
        out["ht_raw_scores"] = raw_scores
        out["ht_normalized_scores"] = norm_scores
        rows.append(out)
        if max_rows > 0 and len(rows) >= max_rows:
            break
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one HT-MNPO player dataset from one reward-model scored file.")
    parser.add_argument("--scored_file", required=True, help="One objective scored file, e.g. skywork=/path/train_skywork.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--normalization", default="none", choices=["minmax", "rank", "none"])
    parser.add_argument("--target_mode", default="reward_gap", choices=["reward_gap", "score_gap", "normalized_gap", "unit"])
    parser.add_argument("--tie_threshold", type=float, default=0.0)
    parser.add_argument("--max_rows", type=int, default=0, help="Optional cap for smoke tests; 0 keeps all rows.")
    args = parser.parse_args()

    objective_name, input_path = parse_named_path(args.scored_file)
    rows = build_rows(
        objective_name=objective_name,
        input_path=input_path,
        normalization=args.normalization,
        target_mode=args.target_mode,
        tie_threshold=args.tie_threshold,
        max_rows=args.max_rows,
    )
    write_jsonl(args.output, rows)
    print(f"Wrote HT-MNPO {objective_name} pairs: {len(rows)} -> {args.output}")


if __name__ == "__main__":
    main()
