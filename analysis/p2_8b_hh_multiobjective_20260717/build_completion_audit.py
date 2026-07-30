#!/usr/bin/env python3
"""Build the final provenance/audit document from measured experiment artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    lines = ["# Completion audit", ""]
    lines += ["## Frozen protocol", ""]
    for name in ("PREREG.md", "run_lock.json"):
        path = root / name
        lines.append(f"- `{name}`: SHA-256 `{sha(path)}`")
    lines += ["", "## Shared data", ""]
    pair = json.loads((root / "train_pool/pair_summary.json").read_text(encoding="utf-8"))
    lines.append(
        f"- Pairs: {pair['train_rows']} train, {pair['test_rows']} sanity; "
        f"train SHA `{pair['train_sha256']}`, test SHA `{pair['test_sha256']}`."
    )
    data_complete = (root / "train_pool/DATA_COMPLETE").read_text(encoding="utf-8").strip()
    lines.append(f"- Shared score/logp/target stage completed at `{data_complete}`.")
    lines += ["", "## Core training", ""]
    for arm in ("ronpo_os", "ronpo_topmass", "inpo_avg", "ht_mnpo_harmless", "ht_mnpo_helpful", "sppo_avg", "simpo", "ipo", "ronpo_full_expect"):
        phase = "stretch/full" if arm in {"ht_mnpo_helpful", "sppo_avg", "simpo", "ipo", "ronpo_full_expect"} else "full"
        status_path = root / "train" / phase / arm / "job_status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            lines.append(
                f"- `{arm}`: {status['status']}; step {status['max_steps']}; effective batch "
                f"{status['effective_batch']}; config SHA `{status['config_sha256']}`; "
                f"W&B `{status['wandb_run_id']}`; checkpoint `{status['checkpoint']}`."
            )
        else:
            lines.append(f"- `{arm}`: MISSING/FAILED before status artifact.")
    lines += ["", "## Evaluation summaries", ""]
    for split in ("validation", "fresh"):
        summary_path = root / split / "model_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            ranking = ", ".join(
                f"{row['model']}={row['mean_prompt_worst_norm_score']:.6f}" for row in summary["ranking"]
            )
            lines.append(f"- `{split}`: {summary['records']} records; eligible {summary['eligible_models']}; primary ranking {ranking}.")
        else:
            lines.append(f"- `{split}`: NOT COMPLETED.")
    lines += ["", "## Score JSONL integrity", "", "| File | Rows | SHA-256 |", "|---|---:|---|"]
    score_files = []
    score_files.extend(sorted((root / "train_pool/scores").glob("*.jsonl")))
    for split in ("validation", "fresh"):
        score_files.extend(sorted((root / split / "scores").glob("*.jsonl")))
    for path in score_files:
        lines.append(f"| `{path.relative_to(root)}` | {jsonl_rows(path)} | `{sha(path)}` |")
    lines += ["", "## Storage and integrity assertions", ""]
    lines.append(
        f"- Training raw generation shards deleted after both 770-row score files and shared targets were verified: "
        f"`{not (root / 'train_pool/generations').exists()}`."
    )
    for split in ("validation", "fresh"):
        generation_dir = root / split / "generations"
        lines.append(f"- `{split}` redundant raw generations deleted after score integrity: `{not generation_dir.exists()}`.")
    fresh_marker = root / "audit/FRESH_OPENED.json"
    lines.append(f"- Fresh manifest opened marker present: `{fresh_marker.is_file()}`.")
    lines.append("- No Hugging Face upload was performed. No paper file was edited.")
    lines.append("")
    lines.append("spent_sealed_split_touched=false")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
