#!/usr/bin/env python3
"""Merge independently scored model subsets into one locked model order."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def rows(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as handle:
        data = [json.loads(line) for line in handle if line.strip()]
    out = {str(row["prompt_id"]): row for row in data}
    if len(out) != len(data):
        raise RuntimeError(f"duplicate prompt ids in {path}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audits", type=Path, nargs="+", required=True)
    p.add_argument("--scores", type=Path, nargs="+", required=True)
    p.add_argument("--final-audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if len(a.audits) != len(a.scores):
        raise RuntimeError("audits and scores must have the same length")
    final_models = json.loads(a.final_audit.read_text())["eligible_models"]
    subsets = []
    for audit_path, score_path in zip(a.audits, a.scores):
        models = json.loads(audit_path.read_text())["eligible_models"]
        subsets.append((models, rows(score_path)))
    prompt_sets = [set(data) for _, data in subsets]
    if not prompt_sets or any(ids != prompt_sets[0] for ids in prompt_sets[1:]):
        raise RuntimeError("scored subsets have different prompt ids")
    merged = []
    for prompt_id in sorted(prompt_sets[0]):
        values = {}
        template = None
        for models, data in subsets:
            row = data[prompt_id]
            template = template or row
            scores = [float(value) for value in row["all_rm_scores"]]
            if len(scores) != len(models):
                raise RuntimeError(f"score/model count mismatch for {prompt_id}")
            for model, value in zip(models, scores):
                if model in values and not math.isclose(values[model], value, rel_tol=1e-5, abs_tol=1e-2):
                    raise RuntimeError(f"inconsistent duplicate score for {prompt_id}/{model}")
                values.setdefault(model, value)
        missing = [model for model in final_models if model not in values]
        if missing:
            raise RuntimeError(f"missing models for {prompt_id}: {missing}")
        row = dict(template)
        row["response_model_names"] = final_models
        row["all_rm_scores"] = [values[model] for model in final_models]
        row.pop("all_generated_responses", None)
        merged.append(row)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": "complete", "records": len(merged), "models": final_models,
        "sha256": hashlib.sha256(a.output.read_bytes()).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()
