#!/usr/bin/env python3
"""Lock the P8 cross-evaluator diagnostic before rescoring existing responses."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite a cross-evaluator protocol lock")
    rows = [json.loads(line) for line in args.responses.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1000:
        raise RuntimeError("P8 diagnostic requires exactly 1,000 existing responses")
    models = rows[0]["response_model_names"]
    if any(row["response_model_names"] != models or len(row["all_generated_responses"]) != len(models) for row in rows):
        raise RuntimeError("response model ordering mismatch")
    payload = {
        "status": "locked_before_scoring",
        "purpose": "post-hoc, reward-model-independent safety diagnostic of fixed P8 Stage-4 responses",
        "selection_impact": "none; the P8 model and table are frozen and this diagnostic cannot select, tune, or retrain a policy",
        "response_file": str(args.responses),
        "response_file_sha256": sha(args.responses),
        "records": len(rows),
        "models": models,
        "evaluators": {
            "llama_guard_3": {
                "model": "meta-llama/Llama-Guard-3-8B",
                "revision": "7327bd9f6efbbe6101dc6cc4736302b3cbb6e425",
                "metric": "logP(safe)-logP(unsafe)",
            },
            "qwen3_guard": {
                "model": "Qwen/Qwen3Guard-Gen-0.6B",
                "revision": "fada3b2f655b89601929198343c94cd2f64d93cc",
                "metric": "generated safe=1, controversial=0.5, unsafe=0",
            },
        },
        "analysis": {
            "all_models_reported": True,
            "metric": "raw mean and paired prompt-level difference versus base",
            "bootstrap_resamples": 2000,
            "bootstrap_seed": 42,
            "normalization": "none",
            "no_new_decode": True,
        },
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spent_sealed_split_touched": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".sha256").write_text(f"{sha(args.output)}  {args.output.name}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
