#!/usr/bin/env python3
"""Write the required clock-aware arm plan from completed smoke artifacts."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-summary", type=Path, required=True)
    parser.add_argument("--deadline", required=True, help="KST ISO datetime")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.smoke_summary.read_text())
    completed = [row for row in summary["arms"] if row["status"] == "completed"]
    failed = [row for row in summary["arms"] if row["status"] != "completed"]
    if failed or len(completed) != 8:
        raise SystemExit("smoke is not a complete 8-arm pass; do not plan W1")
    elapsed = [float(row["elapsed_seconds"]) for row in completed]
    mean_per_step = statistics.mean(value / 20.0 for value in elapsed)
    max_per_step = max(value / 20.0 for value in elapsed)
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    deadline = datetime.fromisoformat(args.deadline).replace(tzinfo=ZoneInfo("Asia/Seoul"))
    waves = 2
    conservative_w1 = max_per_step * 900.0 * waves
    median_w1 = statistics.median(value / 20.0 for value in elapsed) * 900.0 * waves
    available = (deadline - now).total_seconds()
    w1_fits = conservative_w1 <= available
    arm_order = [row["arm"] for row in completed]
    def grad_norm(row: dict) -> float | None:
        if row.get("first_logged_grad_norm") is not None:
            return row["first_logged_grad_norm"]
        log = Path(row["log"])
        if not log.is_file():
            return None
        matches = re.findall(r"['\"]grad_norm['\"]\s*:\s*['\"]?([-+0-9.eE]+)", log.read_text(errors="replace"))
        return float(matches[0]) if matches else None
    lines = [
        "# Clock plan, written after the 20-step smoke and before W1",
        "",
        f"Generated at: `{now.isoformat()}`. Deadline: `{deadline.isoformat()}`. Authorized GPU count: 4.",
        "",
        f"The smoke passed for all {len(completed)} W1 arms. Per-arm elapsed seconds are read from `summary.json`; this includes model loading and dataset formatting, so the full-run estimate is conservative.",
        "",
        f"- Mean elapsed per smoke step: {mean_per_step:.3f} seconds.",
        f"- Slowest elapsed per smoke step: {max_per_step:.3f} seconds.",
        f"- W1 is two 4-GPU waves for 8 arms.",
        f"- Median-proxy W1 estimate: {median_w1 / 3600.0:.2f} GPU-wave wall hours.",
        f"- Conservative W1 estimate: {conservative_w1 / 3600.0:.2f} GPU-wave wall hours.",
        f"- Wall-clock remaining: {available / 3600.0:.2f} hours.",
        f"- Decision: W1 {'fits' if w1_fits else 'does not fit'} before the hard deadline under the conservative smoke proxy.",
        "",
        "## Frozen W1 order",
        "",
        *[f"{index}. `{arm}`" for index, arm in enumerate(arm_order, start=1)],
        "",
        "## Cut order",
        "",
        "W1 is never cut. If W1 uses the available time, W2 is omitted in its pre-registered bottom-up order: fairness LR triple, MNPO, OS entropy-0.85, OS entropy-0.15, uniform control, then top-mass. Every omitted arm is listed later in `CUT.md` with the measured clock time.",
        "",
        "## Smoke evidence",
        "",
        "| Arm | GPU | elapsed s | first logged grad norm |",
        "|---|---:|---:|---:|",
        *[f"| {row['arm']} | {row['gpu']} | {row['elapsed_seconds']:.2f} | {grad_norm(row)} |" for row in completed],
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"w1_fits": w1_fits, "conservative_w1_seconds": conservative_w1, "available_seconds": available}, indent=2))


if __name__ == "__main__":
    main()
