import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def read_generation_file(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def parse_named_paths(values: List[str]) -> List[Tuple[str, str]]:
    out = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected name=path, got {value!r}")
        name, path = value.split("=", 1)
        out.append((name.strip(), path.strip()))
    return out


def prompt_id(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge one-output-per-model decode files into a shared response pool.")
    parser.add_argument("--generations", nargs="+", required=True, help="Named decode outputs, e.g. baseline=... mnpo=...")
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    named_paths = parse_named_paths(args.generations)
    names = [name for name, _ in named_paths]
    by_model: Dict[str, Dict[str, str]] = {}
    prompt_texts: Dict[str, str] = {}

    for name, path in named_paths:
        rows = read_generation_file(path)
        model_rows: Dict[str, str] = {}
        for row in rows:
            prompt = row["prompt"]
            key = prompt_id(prompt)
            prompt_texts[key] = prompt
            model_rows[key] = row["generated_text"]
        by_model[name] = model_rows

    common_keys = set(by_model[names[0]].keys())
    for name in names[1:]:
        common_keys &= set(by_model[name].keys())

    output = []
    for key in sorted(common_keys):
        prompt = prompt_texts[key]
        output.append(
            {
                "prompt_id": key,
                "prompt": prompt,
                "response_model_names": names,
                "all_generated_responses": [by_model[name][key] for name in names],
            }
        )

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(output)} merged prompts for models={names} to {out}")


if __name__ == "__main__":
    main()
