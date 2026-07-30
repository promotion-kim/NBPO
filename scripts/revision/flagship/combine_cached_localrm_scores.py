#!/usr/bin/env python3
"""Combine response-verified cached RM columns into one aligned model panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True,
                        help="label:objective=/path/to/scores.jsonl")
    parser.add_argument("--map", action="append", required=True,
                        help="desired_model=source_label:source_model")
    parser.add_argument("--objective", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    merged = json.loads(args.merged.read_text(encoding="utf-8"))
    desired = list(merged[0]["response_model_names"])
    if any(row["response_model_names"] != desired for row in merged):
        raise RuntimeError("desired generation panel is not aligned")
    mappings = {}
    for value in args.map:
        model, source = value.split("=", 1)
        label, source_model = source.split(":", 1)
        mappings[model] = (label, source_model)
    if set(mappings) != set(desired):
        raise RuntimeError(f"mapping models differ: {set(mappings)} vs {set(desired)}")
    paths = {}
    for value in args.source:
        token, raw_path = value.split("=", 1)
        label, objective = token.split(":", 1)
        paths[(label, objective)] = Path(raw_path)
    provenance = {"merged": str(args.merged), "merged_sha256": sha256(args.merged),
                  "objectives": args.objective, "sources": {}, "mappings": mappings,
                  "response_equality_verified": True, "spent_sealed_split_touched": False}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for objective in args.objective:
        sources = {}
        for label in {value[0] for value in mappings.values()}:
            path = paths.get((label, objective))
            if path is None:
                raise RuntimeError(f"missing source {label}:{objective}")
            rows = load_jsonl(path)
            if len(rows) != len(merged):
                raise RuntimeError(f"row count mismatch: {path}")
            sources[label] = rows
            provenance["sources"][f"{label}:{objective}"] = {"path": str(path), "sha256": sha256(path)}
        output_rows = []
        for index, target in enumerate(merged):
            prompt = str(target["prompt"])
            scores = []
            for model_index, model in enumerate(desired):
                label, source_model = mappings[model]
                source = sources[label][index]
                if str(source["prompt"]) != prompt:
                    raise RuntimeError(f"prompt mismatch at {index}: {label}:{objective}")
                try:
                    source_index = source["response_model_names"].index(source_model)
                except ValueError as exc:
                    raise RuntimeError(f"source model absent: {label}:{objective}:{source_model}") from exc
                if source["all_generated_responses"][source_index] != target["all_generated_responses"][model_index]:
                    raise RuntimeError(f"response mismatch at {index}: {model} from {label}:{source_model}")
                scores.append(float(source["all_rm_scores"][source_index]))
            output_rows.append({"prompt_id": target.get("prompt_id"), "prompt": prompt,
                                "response_model_names": desired,
                                "all_generated_responses": target["all_generated_responses"],
                                "all_rm_scores": scores,
                                "score_cache_composition": {model: mappings[model][0] for model in desired}})
        output = args.output_dir / f"{objective}.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for row in output_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        provenance.setdefault("outputs", {})[objective] = {"path": str(output), "sha256": sha256(output)}
    provenance_path = args.output_dir / "cache_composition_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
