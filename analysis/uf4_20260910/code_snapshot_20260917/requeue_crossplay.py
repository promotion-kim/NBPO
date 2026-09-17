"""Retire the thinking-mode cross-play runs and re-judge under the declared contract.

The declared evaluator contract for this campaign (Appendix: audit protocol) is
the local Qwen3-14B "with thinking disabled ... and a 256-token output limit".
`judge_final_eval.py` passes that flag and parses 99.4% of 190k judgments;
`judge_crossplay_pair.py` did not, so its judge spent the budget on a reasoning
block and parsed 3% at the same limit. Raising the limit to 2048 tokens bought
88-91% but produced numbers under a decoding setting the contract does not
describe, and those are the numbers currently in the cross-play exhibits.

So every thinking-mode run is moved aside under a dotted name -- the aggregator
and the progress counters both already ignore dotted directories -- and all six
pairs are re-judged at the declared 256-token limit with thinking off. The
exhibits go back to unmeasured until the corrected judgments exist; a number
measured under the wrong protocol is worse than a dash.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
QUEUE = ROOT / "jobs" / "queue"
PAIRS = ROOT / "evaluation/crossplay/pairs"
POLICIES = ("base", "nbpo_mse_s42", "fixedref_mse_s42", "util_mse_s42")
GEN_JOB = {p: f"uf4_crossplay_gen_{p}" for p in POLICIES}
PRIORITY = 64


def retire(directory):
    report = directory / "complete.json"
    parsed = None
    if report.exists():
        data = json.loads(report.read_text())
        counts = data.get("status_counts", {})
        total = sum(counts.values()) or 1
        parsed = round(100.0 * counts.get("ok", 0) / total)
        budget = data.get("decoding", {}).get("max_tokens", "unknown")
    else:
        budget = "unknown"
    suffix = f".thinkon_t{budget}_parse{parsed}pct" if parsed is not None else ".thinkon"
    target = Path(str(directory) + suffix)
    if target.exists():
        raise SystemExit(f"refusing to overwrite an existing set-aside run: {target}")
    directory.rename(target)
    return target.name


def main():
    retired = []
    for directory in sorted(PAIRS.iterdir()):
        if directory.is_dir() and "." not in directory.name:
            retired.append(retire(directory))

    removed = []
    for path in sorted(QUEUE.glob("*.json")):
        spec = json.loads(path.read_text())
        job = spec["job_id"]
        if job.startswith("uf4_crossplay_judge_") or job.startswith("uf4_crossplay_aggregate"):
            os.unlink(path)
            removed.append(path.name)

    added = []
    pairs = [(a, b) for i, a in enumerate(POLICIES) for b in POLICIES[i + 1:]]
    judge_jobs = []
    for a, b in pairs:
        job = f"uf4_crossplay_judge_{a}__vs__{b}_think0"
        judge_jobs.append(job)
        spec = {
            "job_id": job, "priority": PRIORITY, "gpus": 1,
            "cwd": str(ROOT / "code"),
            "env": {"PYTHONPATH": "/work/pylibs_eval", "HF_HUB_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                    "OMP_NUM_THREADS": "4", "WANDB_MODE": "disabled"},
            "depends_on": sorted({GEN_JOB[a], GEN_JOB[b]}),
            "command": ["python3", str(ROOT / "code/judge_crossplay_pair.py"),
                        "--policy-a", a, "--policy-b", b,
                        "--pool-a", str(ROOT / f"pools/crossplay_{a}"),
                        "--pool-b", str(ROOT / f"pools/crossplay_{b}")],
            "timeout_s": 21600,
            "artifacts": [str(PAIRS / f"{a}__vs__{b}/complete.json")]}
        path = QUEUE / f"{PRIORITY}_{job}.json"
        path.write_text(json.dumps(spec, indent=2) + "\n")
        added.append(path.name)

    spec = {"job_id": "uf4_crossplay_aggregate_think0", "priority": PRIORITY, "gpus": 0,
            "cwd": str(ROOT / "code"), "env": {"PYTHONPATH": "/work/pylibs_eval"},
            "depends_on": judge_jobs,
            "command": ["python3", str(ROOT / "code/aggregate_crossplay.py")],
            "timeout_s": 7200,
            "artifacts": [str(ROOT / "analysis/crossplay_summary.json")]}
    path = QUEUE / f"{PRIORITY}_uf4_crossplay_aggregate_think0.json"
    path.write_text(json.dumps(spec, indent=2) + "\n")
    added.append(path.name)

    print("retired runs:")
    for name in retired:
        print("  " + name)
    print(f"removed {len(removed)} stale judge/aggregate specs")
    print("added:")
    for name in added:
        print("  " + name)


if __name__ == "__main__":
    main()
