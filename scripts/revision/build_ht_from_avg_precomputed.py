from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from datasets import load_from_disk


USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)


def _extract_user_prompt(prompt: Any) -> str:
    if not isinstance(prompt, str):
        return str(prompt)
    match = USER_RE.search(prompt)
    if match:
        return match.group(1).strip()
    assistant_marker = "<|im_start|>assistant"
    if assistant_marker in prompt:
        return prompt.split(assistant_marker, 1)[0].strip()
    return prompt.strip()


def _messages(prompt: str, response: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def _scores(row: dict[str, Any], objective: str) -> list[float]:
    normalized = row.get("normalized_objective_scores")
    if isinstance(normalized, dict) and objective in normalized:
        return [float(value) for value in normalized[objective]]
    raw = row.get("objective_scores")
    if isinstance(raw, dict) and objective in raw:
        values = [float(value) for value in raw[objective]]
        lo, hi = min(values), max(values)
        if abs(hi - lo) < 1e-12:
            return [0.5 for _ in values]
        return [(value - lo) / (hi - lo) for value in values]
    raise KeyError(f"missing objective scores for {objective!r}")


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda idx: values[idx])


def _argmin(values: list[float]) -> int:
    return min(range(len(values)), key=lambda idx: values[idx])


def build_split(pref_dir: Path, split: str, objective: str, output: Path, max_rows: int) -> int:
    dataset = load_from_disk(str(pref_dir))[split]
    count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in dataset:
            responses = row.get("all_generated_responses")
            if not isinstance(responses, list) or len(responses) < 2:
                continue
            scores = _scores(row, objective)
            if len(scores) != len(responses):
                continue
            chosen_idx = _argmax(scores)
            rejected_idx = _argmin(scores)
            if chosen_idx == rejected_idx:
                continue
            prompt = _extract_user_prompt(row.get("prompt"))
            target = float(scores[chosen_idx] - scores[rejected_idx])
            if abs(target) < 1e-12:
                continue
            out = {
                key: value
                for key, value in row.items()
                if not key.endswith("_logps") and "_logps_" not in key
            }
            out["prompt"] = [{"role": "user", "content": prompt}]
            out["chosen"] = _messages(prompt, str(responses[chosen_idx]))
            out["rejected"] = _messages(prompt, str(responses[rejected_idx]))
            out["chosen_index"] = int(chosen_idx)
            out["rejected_index"] = int(rejected_idx)
            out["pair_source"] = "ht_mnpo_player_objective_from_avg_precompute"
            out["ht_target"] = target
            out["ht_objective_name"] = objective
            out["ht_objective_gap"] = target
            out["ht_normalized_gap"] = target
            out["ht_target_mode"] = "normalized_gap"
            out["ht_score_normalization"] = "precomputed_minmax"
            out["ht_normalized_scores"] = scores
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1
            if max_rows > 0 and count >= max_rows:
                break
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pref-dir", required=True)
    parser.add_argument("--objective", required=True, choices=["skywork", "athene", "armo"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    pref_dir = Path(args.pref_dir)
    output_dir = Path(args.output_dir)
    train_n = build_split(
        pref_dir,
        "train",
        args.objective,
        output_dir / f"train_ht_{args.objective}.jsonl",
        args.max_rows,
    )
    test_n = build_split(
        pref_dir,
        "test",
        args.objective,
        output_dir / f"test_ht_{args.objective}.jsonl",
        args.max_rows,
    )
    summary = {
        "source": str(pref_dir),
        "objective": args.objective,
        "train_examples": train_n,
        "test_examples": test_n,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
