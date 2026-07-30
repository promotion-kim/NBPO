#!/usr/bin/env python3
"""Append a narrowly scoped scheduling fix to the ARC capability audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    timestamp = datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")
    text = (
        f"## {timestamp}\n\n"
        "The first background finalizer command had shell-quoting syntax invalid before its loop began. "
        "It did not start a model, score a request, or modify an ARC result. The immutable capability lock "
        "and the completed worker artifacts are preserved. Finalization is re-queued through "
        "`wait_finalize_p8_stage4_arc_challenge.sh`, which only waits for all nine pre-locked result files "
        "and invokes the unchanged JSON aggregator.\n\n"
    )
    path = args.output / "fix_log.md"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


if __name__ == "__main__":
    main()
