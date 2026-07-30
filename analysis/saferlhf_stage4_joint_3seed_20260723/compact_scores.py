#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def rows(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = {
        "beaver_helpfulness": "beaver_helpfulness.jsonl",
        "beaver_harmlessness": "beaver_harmlessness.jsonl",
        "independent_quality": "independent_quality.jsonl",
        "independent_safety": "independent_safety_merged.jsonl",
    }
    loaded = {name: list(rows(args.score_dir / filename)) for name, filename in files.items()}
    reference = loaded["beaver_helpfulness"]
    prompt_ids = [row["prompt_id"] for row in reference]
    model_names = reference[0]["response_model_names"]
    for name, data in loaded.items():
        assert [row["prompt_id"] for row in data] == prompt_ids, name
        assert all(row["response_model_names"] == model_names for row in data), name
        assert all(len(row["all_rm_scores"]) == len(model_names) for row in data), name
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for index, prompt_id in enumerate(prompt_ids):
            item = {
                "prompt_id": prompt_id,
                "model_names": model_names,
                "scores": {
                    name: data[index]["all_rm_scores"] for name, data in loaded.items()
                },
            }
            handle.write(json.dumps(item, separators=(",", ":")) + "\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"rows": len(prompt_ids), "sha256": digest, "output": str(args.output)}))


if __name__ == "__main__":
    main()
