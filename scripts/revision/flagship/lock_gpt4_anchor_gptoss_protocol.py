#!/usr/bin/env python3
"""Freeze the GPT-4-anchor/open-weight-judge protocol before any judgments."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


JUDGE_MODEL = "openai/gpt-oss-120b"
JUDGE_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
    model_names = manifest["model_names"]
    if len(model_names) != 11:
        raise RuntimeError("protocol requires exactly 11 frozen flagship models")
    expected = {
        "alpaca_eval_2": 805 * len(model_names) * 2,
        "arena_hard_v0.1": 500 * len(model_names) * 2,
        "mt_bench": 80 * len(model_names) * 2,
    }
    protocol = {
        "locked_at_kst": dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(),
        "status": "FROZEN_BEFORE_JUDGING",
        "scope": "zero-cost GPT-4-reference open-weight-judge proxy",
        "official_score_disclaimer": (
            "Not official AlpacaEval 2, Arena-Hard, or MT-Bench scores because the closed judge is replaced "
            "by a pinned open-weight judge. No claim of validated equivalence to GPT-5.4-mini."
        ),
        "selection_policy": "No model, prompt, rubric, decode, parse, or aggregation changes after rankings are observed.",
        "judge": {
            "model": JUDGE_MODEL, "revision": JUDGE_REVISION, "license": "Apache-2.0",
            "reasoning_effort": "low", "temperature": 0.0, "top_p": 1.0,
            "max_new_tokens": 4096, "max_model_len": 16384, "seed": 42,
            "dtype": "auto", "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.86, "trust_remote_code": False,
        },
        "benchmarks": {
            "alpaca_eval_2": {
                "reference": "public gpt-4-turbo / gpt4_1106_preview outputs",
                "rubric": "EvalScope 1.0.2 AlpacaEval adapter verbatim",
                "orders": ["reference_first", "candidate_first"],
                "primary_proxy": "symmetric raw win rate; official LC transform also reported on reference-first annotations when available",
                "tie_policy": "no explicit tie in the frozen EvalScope rubric; inconsistent position swaps average to 0.5",
            },
            "arena_hard_v0.1": {
                "reference": "public gpt-4-0314 outputs",
                "rubric": "EvalScope 1.0.2 Arena-Hard five-grade adapter verbatim",
                "orders": ["reference_first", "candidate_first"],
                "severity_weights": {"significant": 3, "slight": 1, "tie": 1},
                "primary_proxy": "anchor win rate from position-swapped severity-weighted battles",
            },
            "mt_bench": {
                "reference": "official FastChat GPT-4 answers for math/reasoning/coding only",
                "rubric": "FastChat single-v1/single-math-v1 and multi-turn variants verbatim",
                "primary_proxy": "mean absolute 1-10 score across both turns and 80 questions",
            },
        },
        "bootstrap": {"unit": "prompt", "resamples": 2000, "seed": 42, "ci": 0.95},
        "models": model_names,
        "model_rows": manifest["models"],
        "references": manifest["references"],
        "expected_judgments": expected,
        "expected_total_judgments": sum(expected.values()),
        "source_response_counts": manifest["response_counts"],
        "pre_ranking_amendment": {
            "supersedes": "protocol_lock.json (attempt 1)",
            "reason": (
                "Attempt 1 produced no aggregate/ranking. Harmony final labels required a deterministic parser update, "
                "and Arena-Hard judgments were truncated before verdict at 1024 tokens. The official Arena-Hard v0.1 "
                "configuration uses max_tokens=4096, applied symmetrically to every model in this full rerun."
            ),
            "attempt1_preserved": True,
        },
    }
    protocol["configuration_sha256"] = sha256_text(json.dumps(protocol, sort_keys=True))
    serialized = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        for key in ("locked_at_kst", "configuration_sha256"):
            previous.pop(key, None)
        comparable = dict(protocol)
        for key in ("locked_at_kst", "configuration_sha256"):
            comparable.pop(key, None)
        if previous != comparable:
            raise RuntimeError("existing protocol lock differs; refusing to overwrite")
        print(args.output, flush=True)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, flush=True)


if __name__ == "__main__":
    main()
