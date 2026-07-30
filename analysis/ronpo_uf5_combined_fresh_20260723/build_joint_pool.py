#!/usr/bin/env python3
import argparse
import json
import re
import unicodedata
from pathlib import Path


def load(path):
    rows = json.loads(Path(path).read_text())
    prompts, responses = [], []
    for row in rows:
        ys = row.get("all_generated_responses")
        text = row.get("generated_text")
        if text is None and ys:
            text = ys[0]
        if text is None:
            raise ValueError(f"missing response in {path}")
        prompts.append(row["prompt"])
        responses.append(str(text))
    return prompts, responses


def norm(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-manifest", type=Path, required=True)
    p.add_argument("--models", nargs="+", required=True, help="tag=label=path")
    p.add_argument("--index-aligned-tags", nargs="*", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    a = p.parse_args()
    specs = []
    for spec in a.models:
        tag, rest = spec.split("=", 1)
        label, path = rest.rsplit("=", 1)
        specs.append((tag, label, path))
    manifest = json.loads(a.prompt_manifest.read_text())["prompts"]
    tables, align_audit = {}, {}
    for tag, _, path in specs:
        prompts, responses = load(path)
        if tag in a.index_aligned_tags:
            if len(prompts) != len(manifest):
                raise RuntimeError(f"{tag}: {len(prompts)} rows, expected {len(manifest)}")
            normalized_matches = sum(norm(x) == norm(y) for x, y in zip(manifest, prompts))
            if normalized_matches < int(0.98 * len(manifest)):
                raise RuntimeError(f"{tag}: only {normalized_matches}/{len(manifest)} index matches")
            tables[tag] = dict(zip(manifest, responses))
            align_audit[tag] = {
                "mode": "index",
                "rows": len(prompts),
                "normalized_prompt_matches": normalized_matches,
                "nonmatches": len(manifest) - normalized_matches,
                "reason": "RMOD preserves manifest order; long prompts are tokenizer-truncated in its output metadata.",
            }
        else:
            tables[tag] = dict(zip(prompts, responses))
            align_audit[tag] = {"mode": "exact_prompt"}
    missing = {tag: [x for x in manifest if x not in table] for tag, table in tables.items()}
    if any(missing.values()):
        raise RuntimeError({k: len(v) for k, v in missing.items()})
    rows = [
        {"prompt": prompt, "all_generated_responses": [tables[tag][prompt] for tag, _, _ in specs]}
        for prompt in manifest
    ]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(rows, ensure_ascii=False) + "\n")
    audit = {
        "prompt_count": len(rows),
        "model_count": len(specs),
        "models": [{"tag": tag, "label": label, "path": path} for tag, label, path in specs],
        "response_order": [tag for tag, _, _ in specs],
        "alignment": align_audit,
    }
    a.audit.write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
