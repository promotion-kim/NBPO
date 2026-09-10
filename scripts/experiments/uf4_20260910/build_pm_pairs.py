"""Materialize the UF-4 preference-model pairs from the released annotations.

Each prompt group contributes the C(4,2)=6 unordered pairs of its four released
completions. For every criterion the target is 1 / 0 / 0.5 for A better / B
better / tied, and a missing rating on either side masks that criterion for that
pair. A masked criterion contributes nothing to the loss and nothing to its
denominator; it is never imputed as a tie and never replaced by the overall
score.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from collections import Counter
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
SNAPSHOT = Path("/work/hf_cache/hub/datasets--openbmb--UltraFeedback/snapshots/"
                "40b436560ca83a8dba36114c22ab3c66e43f6d5e")
OBJECTIVES = ("instruction_following", "truthfulness", "honesty", "helpfulness")

import re


def parse_rating(annotation):
    if not isinstance(annotation, dict):
        return None
    raw = annotation.get("Rating")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        value = float(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        if not re.fullmatch(r"[1-5](\.0+)?", text):
            return None
        value = float(text)
    else:
        return None
    return int(value) if 1 <= value <= 5 and float(value).is_integer() else None


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_index():
    """line_no -> raw row, per source file, read once."""
    index = {}
    for source in ("ultrachat", "sharegpt", "evol_instruct"):
        rows = {}
        with (SNAPSHOT / f"{source}.jsonl").open() as stream:
            for line_no, line in enumerate(stream):
                line = line.strip()
                if line:
                    rows[line_no] = line
        index[source] = rows
    return index


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", default="v1")
    ap.add_argument("--out-name", default="v1")
    args = ap.parse_args()
    start = time.monotonic()
    split_dir = ROOT / "splits" / args.splits
    out = ROOT / "data" / f"pm_pairs_{args.out_name}"
    out.mkdir(parents=True, exist_ok=False)
    raw = source_index()

    report = {"splits_dir": str(split_dir), "objectives": list(OBJECTIVES),
              "target_convention": "1 A>B, 0 A<B, 0.5 tie; missing rating on either side masks that criterion",
              "pairs_per_prompt": 6}
    for split in ("pm_train", "pm_dev"):
        path = split_dir / f"{split}.jsonl"
        out_path = out / f"{split}.jsonl"
        n_prompts = n_pairs = 0
        labeled = Counter()
        tied = Counter()
        response_lengths = []
        with path.open() as stream, out_path.open("x") as sink:
            for line in stream:
                group = json.loads(line)
                row = json.loads(raw[group["source"]][group["line_no"]])
                if row["instruction"] != group["instruction"]:
                    raise ValueError(f"{split}: source row {group['line_no']} no longer matches the split")
                completions = row["completions"]
                responses, ratings, models = [], [], []
                for completion in completions:
                    responses.append(completion["response"])
                    models.append(completion.get("model"))
                    annotations = completion.get("annotations") or {}
                    ratings.append({obj: parse_rating(annotations.get(obj)) for obj in OBJECTIVES})
                response_lengths.extend(len(text) for text in responses)
                n_prompts += 1
                for a, b in itertools.combinations(range(len(responses)), 2):
                    targets, mask = {}, {}
                    for obj in OBJECTIVES:
                        ra, rb = ratings[a][obj], ratings[b][obj]
                        if ra is None or rb is None:
                            targets[obj], mask[obj] = None, False
                            continue
                        targets[obj] = 1.0 if ra > rb else (0.0 if ra < rb else 0.5)
                        mask[obj] = True
                        labeled[obj] += 1
                        if ra == rb:
                            tied[obj] += 1
                    sink.write(json.dumps({
                        "prompt_id": group["prompt_id"], "source": group["source"],
                        "instruction": row["instruction"],
                        "response_a": responses[a], "response_b": responses[b],
                        "model_a": models[a], "model_b": models[b],
                        "candidate_a": a, "candidate_b": b,
                        "rating_a": {obj: ratings[a][obj] for obj in OBJECTIVES},
                        "rating_b": {obj: ratings[b][obj] for obj in OBJECTIVES},
                        "target": targets, "mask": mask}, ensure_ascii=False) + "\n")
                    n_pairs += 1
        report[split] = {
            "path": str(out_path), "sha256": file_hash(out_path),
            "n_prompts": n_prompts, "n_pairs": n_pairs,
            "labeled_pairs_per_objective": dict(labeled),
            "tied_pairs_per_objective": dict(tied),
            "tie_fraction_per_objective": {obj: round(tied[obj] / labeled[obj], 4)
                                           for obj in OBJECTIVES if labeled[obj]},
            "masked_pairs_per_objective": {obj: n_pairs - labeled[obj] for obj in OBJECTIVES},
            "response_chars_mean": round(sum(response_lengths) / len(response_lengths), 1),
            "response_chars_p99": sorted(response_lengths)[int(0.99 * len(response_lengths))]}
        print(json.dumps({"split": split, **{k: v for k, v in report[split].items()
                                             if k != "path"}}), flush=True)
    report["seconds"] = time.monotonic() - start
    report["source_sha256"] = file_hash(__file__)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
