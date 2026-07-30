#!/usr/bin/env python3
"""Freeze the full Arena verdict-only adaptation before any adapted judgments."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent = json.loads(args.parent_protocol.read_text(encoding="utf-8"))
    protocol = {
        "status": "FROZEN_BEFORE_ARENA_ADAPTED_JUDGING",
        "locked_at_kst": dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(),
        "parent_protocol_sha256": parent["configuration_sha256"],
        "reason": (
            "Before any aggregate/ranking, the full 11,000-call Arena branch is rerun symmetrically. The official-rubric "
            "attempt often omitted the verdict, the 512-token verdict-only attempt had a truncated trace, and the first "
            "2,048-token attempt returned bare A for both positions of one prompt. No bare label is mapped to a severity "
            "class. The final full rerun explicitly lists all five allowed bracketed strings in both system and user text."
        ),
        "official_score_disclaimer": parent["official_score_disclaimer"],
        "judge": dict(parent["judge"], max_new_tokens=2048),
        "models": parent["models"],
        "model_rows": parent["model_rows"],
        "references": {"arena_hard_v0.1": parent["references"]["arena_hard_v0.1"]},
        "benchmark": {
            "name": "arena_hard_v0.1",
            "calls": 500 * len(parent["models"]) * 2,
            "orders": ["reference_first", "candidate_first"],
            "rubric": (
                "Open-weight format adaptation: same EvalScope criteria and five labels, but omit the request to "
                "generate an independent answer and require exactly one of five bracketed verdict labels in both system "
                "and user text. Applied to all models and both positions."
            ),
            "labels": ["A>>B", "A>B", "A=B", "B>A", "B>>A"],
            "severity_weights": {"significant": 3, "slight": 1, "tie": 1},
        },
    }
    protocol["configuration_sha256"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite existing Arena adaptation lock")
    args.output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == "__main__":
    main()
