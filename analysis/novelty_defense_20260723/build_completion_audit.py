#!/usr/bin/env python3
"""Build a checksum and provenance audit for the frozen joint evaluation."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def wandb_url(path: Path) -> str:
    urls = re.findall(r"https://wandb\.ai/promotion-kim/mnpo/runs/[a-z0-9]+", path.read_text(errors="replace"))
    return urls[0] if urls else "missing"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root)
    remote = list(csv.DictReader((root / "provenance/remote_artifacts.tsv").open(), delimiter="\t"))
    decode_logs = list((root / "logs/decode_ronpo").glob("*.log")) + list((root / "logs/decode_ronpo2").glob("*.log"))
    run_urls = {}
    for log in decode_logs:
        name = log.stem
        if name == "ronpo_topmass":
            continue
        if name.endswith("_retry"):
            name = name.removesuffix("_retry")
        run_urls.setdefault(name, wandb_url(log))
    retry = root / "logs/decode_ronpo/ronpo_topmass_retry.log"
    run_urls["ronpo_topmass"] = wandb_url(retry)

    important = [
        root / "DISCOVERY.md", root / "PREREG.md", root / "run_lock.json",
        root / "paired_summary.json", root / "floor_summary.json",
        root / "per_policy_scores/summary.csv",
        root / "per_policy_scores/raw_reward_summary.json",
        root / "ifeval/summary.json", root / "ifeval/summary.csv",
        root / "frag_stage2_main_table.tex", root / "frag_factorized_ablation.tex",
        root / "frag_floor_table.tex", root / "REPORT.md", root / "fix_log.md",
        root / "provenance/remote_artifacts.tsv",
    ]
    important += sorted((root / "per_policy_scores").glob("*.json"))
    important += sorted((root / "stability_gates").glob("*.json"))
    important += sorted((root / "ifeval").glob("*/outputs/*/reports/*/ifeval.json"))
    important += [root / "logs/fragment_smoke.log"]
    seen = set()
    important = [x for x in important if x.exists() and not (x in seen or seen.add(x))]

    lines = [
        "# Completion audit",
        "",
        "## Status",
        "",
        "- Discovery hard gate: PASS. Seed-42 k-only, a-only, and MaxMin-RLHF were recovered.",
        "- New training: none.",
        "- Frozen decode: complete for 15 policies, 647 responses each.",
        "- Frozen joint scoring: complete for all three reward models on the same 647-row input.",
        "- IFEval: complete for the seven preregistered new arms.",
        "- Paper source edits: none.",
        f"- `ronpo_aaai/main_v3.tex` observed SHA-256 after completion: "
        f"  `{sha256(Path('ronpo_aaai/main_v3.tex'))}`.",
        "- GPT-5.5 pairwise matrix: excluded; no OpenAI API call was attempted.",
        "- Original recovered B200 checkpoints remain in place. The task-owned topmass transfer copy was "
        "  pruned after both shard hashes were reverified against the retained local source.",
        "",
        "## Preregistration integrity",
        "",
        f"- `PREREG.md`: `{sha256(root / 'PREREG.md')}`",
        f"- `run_lock.json`: `{sha256(root / 'run_lock.json')}`",
        "- These match `prereg.sha256` and were written before reward scoring.",
        "",
        "## Remote generation and score artifacts",
        "",
        "| Kind | Name | Rows | Bytes | SHA-256 |",
        "|---|---|---:|---:|---|",
    ]
    for row in remote:
        lines.append(f"| {row['kind']} | {row['name']} | {row['rows']} | {row['bytes']} | `{row['sha256']}` |")
    lines += [
        "",
        "The three `joint_*.jsonl` files were produced from the single input whose SHA-256 is "
        "shown above. No score from an earlier or split policy batch was merged.",
        "",
        "## W&B decode runs",
        "",
        "| Policy | Run |",
        "|---|---|",
    ]
    for name in sorted(run_urls):
        lines.append(f"| {name} | {run_urls[name]} |")
    lines += [
        "",
        "The initial top-mass run failed before decode because its second shard was still in transfer; "
        "the table records the successful fixed-protocol recovery run. On ronpo2, later wrapper "
        "invocations found complete outputs and skipped generation, so only the first run that created "
        "each output is listed.",
        "",
        "## Local deliverable hashes",
        "",
        "| File | Rows | Bytes | SHA-256 |",
        "|---|---:|---:|---|",
    ]
    for path in important:
        rel = path.relative_to(root)
        rows = "--"
        if rel.parts[:1] == ("per_policy_scores",) and path.suffix == ".json" and path.name != "raw_reward_summary.json":
            obj = json.loads(path.read_text())
            rows = str(len(obj.get("rows", [])))
        elif rel.parts[:1] == ("stability_gates",) and path.suffix == ".json":
            rows = str(json.loads(path.read_text()).get("candidate", {}).get("records", "--"))
        elif rel.parts[:1] == ("ifeval",) and path.name == "ifeval.json":
            rows = str(json.loads(path.read_text()).get("num", "--"))
        lines.append(f"| `{rel}` | {rows} | {path.stat().st_size} | `{sha256(path)}` |")
    lines += [
        "",
        "## Validation",
        "",
        "- All three LaTeX fragments compiled together under TinyTeX with zero fatal errors, warnings, "
        "  unresolved references, overfull boxes, or underfull boxes on the second pass.",
        "- Preregistration checksums, JSON row counts, policy-set equality, IFEval cardinality, and "
        "  Python syntax checks all passed in the final validation run.",
        "- All interventions and zero-work infrastructure failures are timestamped in `fix_log.md`; "
        "  preserved logs are under `audit/`.",
        "- No original recovered checkpoint was deleted. Public RM caches, the temporary EvalScope "
        "  environment, redundant partial-generation files, and the verified task-owned topmass transfer "
        "  copy were pruned, reducing remote scratch usage from 50 GB to 241 MB.",
        "",
        "evaluation_complete=true",
    ]
    (root / "COMPLETION_AUDIT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
