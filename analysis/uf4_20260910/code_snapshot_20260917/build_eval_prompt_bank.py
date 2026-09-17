"""Collect every evaluation prompt UF-4 must not train on, in one normalized bank.

Decontamination happens BEFORE the split, so this bank is an input to the split
and never a filter applied afterwards. A source that cannot be read is recorded
as unavailable with its reason: an unread benchmark is a stated limit of the
decontamination, never a silently empty set.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
LOCAL = Path("/work/nbpo_downstream_local_v1/data")


def normalize(text: str) -> str:
    """The one prompt normalization every hash and shingle in UF-4 uses."""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path):
    with open(path) as stream:
        for line in stream:
            line = line.strip()
            if line:
                yield json.loads(line)


def local_sources():
    yield "ifeval", (row["prompt"] for row in read_jsonl(LOCAL / "ifeval.jsonl"))
    yield "gsm8k_test", (row["question"] for row in read_jsonl(LOCAL / "gsm8k_test.jsonl"))
    yield "alpaca_eval", (row["instruction"] for row in read_jsonl(LOCAL / "alpaca_eval.jsonl"))
    yield "arena_hard", (row["prompt"] for row in read_jsonl(LOCAL / "arena_hard.jsonl"))
    yield "harmbench", (row["Behavior"] for row in read_jsonl(LOCAL / "harmbench_behaviors.jsonl"))
    xstest = Path("/work/nbpo_repair_20260909/responses/mse_short_primary_v1/xstest.jsonl")
    yield "xstest", (row["prompt"] for row in read_jsonl(xstest))


def hub_sources():
    """Multiple-choice benchmarks, fetched once and pinned by row count."""
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("HF_DATASETS_OFFLINE", None)
    os.environ["HF_HOME"] = "/work/hf_cache"
    from datasets import load_dataset
    specs = [
        ("mmlu", lambda: load_dataset("cais/mmlu", "all", split="test"), "question"),
        ("arc_challenge", lambda: load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"), "question"),
        ("hellaswag", lambda: load_dataset("Rowan/hellaswag", split="validation"), "ctx"),
    ]
    for name, loader, column in specs:
        try:
            data = loader()
            yield name, (row[column] for row in data), None
        except Exception as error:                       # noqa: BLE001
            yield name, iter(()), f"{type(error).__name__}: {error}"[:300]


def main():
    out = ROOT / "data" / "eval_prompt_bank.jsonl"
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    counts, unavailable = {}, {}
    seen = {}
    with out.open("x") as stream:
        for name, prompts in local_sources():
            n = 0
            for prompt in prompts:
                norm = normalize(prompt)
                if not norm:
                    continue
                n += 1
                key = sha(norm)
                if key in seen:
                    seen[key].append(name)
                    continue
                seen[key] = [name]
                stream.write(json.dumps({"benchmark": name, "norm_sha256": key,
                                         "norm_text": norm}) + "\n")
            counts[name] = n
        for name, prompts, error in hub_sources():
            if error:
                unavailable[name] = error
                counts[name] = 0
                continue
            n = 0
            for prompt in prompts:
                norm = normalize(prompt)
                if not norm:
                    continue
                n += 1
                key = sha(norm)
                if key in seen:
                    seen[key].append(name)
                    continue
                seen[key] = [name]
                stream.write(json.dumps({"benchmark": name, "norm_sha256": key,
                                         "norm_text": norm}) + "\n")
            counts[name] = n
    report = {"per_benchmark_rows_read": counts,
              "unique_normalized_prompts": len(seen),
              "unavailable_benchmarks": unavailable,
              "decontamination_limit": ("Only these evaluation prompt sets are checked. "
                                        "Pretraining-corpus overlap cannot be checked here and "
                                        "remains a stated limitation."),
              "bank_path": str(out),
              "bank_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / "provenance" / "eval_prompt_bank.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if unavailable:
        print("NOTE: some benchmarks were unavailable; see unavailable_benchmarks", file=sys.stderr)


if __name__ == "__main__":
    main()
