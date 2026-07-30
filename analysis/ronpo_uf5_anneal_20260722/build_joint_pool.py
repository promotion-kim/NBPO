#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load(path):
    rows = json.loads(Path(path).read_text())
    out = {}
    for row in rows:
        ys = row.get("all_generated_responses")
        text = row.get("generated_text")
        if text is None and ys:
            text = ys[0]
        if text is None:
            raise ValueError(f"missing response in {path}")
        out[row["prompt"]] = str(text)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-manifest", type=Path, required=True)
    p.add_argument("--models", nargs="+", required=True, help="tag=label=path")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    a = p.parse_args()
    # Labels may contain '=' (for example, "RMOD K=16").  The tag is before
    # the first delimiter and the path is after the last delimiter.
    specs = []
    for spec in a.models:
        tag, rest = spec.split("=", 1)
        label, path = rest.rsplit("=", 1)
        specs.append((tag, label, path))
    tables = {tag: load(path) for tag, _, path in specs}
    prompts = json.loads(a.prompt_manifest.read_text())["prompts"]
    missing = {tag: [x for x in prompts if x not in table] for tag, table in tables.items()}
    if any(missing.values()):
        raise RuntimeError({k: len(v) for k, v in missing.items()})
    rows = [
        {"prompt": prompt, "all_generated_responses": [tables[tag][prompt] for tag, _, _ in specs]}
        for prompt in prompts
    ]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(rows, ensure_ascii=False) + "\n")
    audit = {
        "prompt_count": len(rows),
        "model_count": len(specs),
        "models": [{"tag": tag, "label": label, "path": path} for tag, label, path in specs],
        "response_order": [tag for tag, _, _ in specs],
    }
    a.audit.write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
