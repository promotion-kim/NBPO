#!/usr/bin/env python3
"""Run preregistered RM resolution/conflict diagnostics on four authorized GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


SKYWORK = ("Skywork/Skywork-Reward-V2-Llama-3.1-8B", "cba2f842f3f1af2f1b2f0d35e794d789976390c5")
ATHENE = ("Nexusflow/Athene-RM-8B", "cdf428f7b52a323b6cf4e9803e5bcba9f1fb5a59")


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(bool(line.strip()) for line in path.open(encoding="utf-8"))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def env(project: Path, cache: Path, gpu: int) -> dict[str, str]:
    value = os.environ.copy()
    value.update({
        "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(project),
        "HF_HOME": str(cache), "HF_HUB_CACHE": str(cache),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "TORCH_CUDNN_SDPA_ENABLED": "0",
    })
    return value


def command(project: Path, python: str, kind: str, input_path: Path, output_root: Path, cache: Path) -> tuple[list[str], list[Path]]:
    if kind == "armo":
        root = output_root / "armo"
        return ([python, str(project / "scripts/revision/flagship/score_armo_primary_heads.py"),
                 "--input-file", str(input_path), "--output-dir", str(root),
                 "--cache-dir", str(cache), "--batch-size", "8", "--sample-batch-size", "4",
                 "--max-seq-length", "4096", "--local-files-only"],
                [root / f"{name}.jsonl" for name in ("helpfulness", "safety", "conciseness")])
    model, revision = SKYWORK if kind == "skywork" else ATHENE
    output = output_root / f"{kind}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [python, "-u", "-m", f"on_policy_data_gen.rm_{kind}",
           "--input_file", str(input_path), "--output_file", str(output),
           "--cache_dir", str(cache), "--model_name", model, "--revision", revision,
           "--local_files_only", "--batch_size", "16", "--sample_batch_size", "8"]
    if kind == "skywork":
        cmd.extend(["--max_seq_length", "4096", "--attn_implementation", "sdpa"])
    return cmd, [output]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--armo-cache", type=Path, required=True)
    parser.add_argument("--general-rm-cache", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = args.output_dir / "logs"
    logs.mkdir(exist_ok=True)

    specs = [
        (0, "conflict", "armo", args.armo_cache),
        (1, "conflict", "skywork", args.general_rm_cache),
        (2, "conflict", "athene", args.general_rm_cache),
    ]
    jobs = []
    reused = []
    for gpu, split, kind, cache in specs:
        input_path = args.input_dir / f"{split}_pool.json"
        output_root = args.output_dir / split
        cmd, expected = command(args.project, args.python, kind, input_path, output_root, cache)
        if all(count_jsonl(path) == 512 for path in expected):
            reused.append({"gpu": gpu, "split": split, "kind": kind,
                           "status": "reused_complete", "outputs": [str(path) for path in expected]})
            continue
        handle = (logs / f"{split}_{kind}.log").open("a", encoding="utf-8")
        process = subprocess.Popen(cmd, cwd=args.project, env=env(args.project, cache, gpu), stdout=handle, stderr=subprocess.STDOUT)
        jobs.append((gpu, split, kind, process, handle, cmd, expected))

    # GPU 3 handles all three smaller resolution jobs sequentially in one worker.
    resolution_commands = []
    resolution_expected = []
    for kind, cache in (("armo", args.armo_cache), ("skywork", args.general_rm_cache), ("athene", args.general_rm_cache)):
        cmd, expected = command(args.project, args.python, kind,
                                args.input_dir / "resolution_controls.json",
                                args.output_dir / "resolution", cache)
        if not all(count_jsonl(path) == 128 for path in expected):
            resolution_commands.append((cmd, cache, kind))
        else:
            reused.append({"gpu": 3, "split": "resolution", "kind": kind,
                           "status": "reused_complete", "outputs": [str(path) for path in expected]})
        resolution_expected.extend(expected)
    if resolution_commands:
        worker = args.output_dir / "run_resolution_worker.sh"
        lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
        for cmd, cache, kind in resolution_commands:
            quoted = " ".join(subprocess.list2cmdline([part]) for part in cmd)
            lines.append(f"HF_HOME={subprocess.list2cmdline([str(cache)])} HF_HUB_CACHE={subprocess.list2cmdline([str(cache)])} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH={subprocess.list2cmdline([str(args.project)])} TOKENIZERS_PARALLELISM=false TORCH_CUDNN_SDPA_ENABLED=0 {quoted}")
        worker.write_text("\n".join(lines) + "\n", encoding="utf-8")
        handle = (logs / "resolution_all.log").open("a", encoding="utf-8")
        process = subprocess.Popen(["bash", str(worker)], cwd=args.project,
                                   env={**os.environ, "CUDA_VISIBLE_DEVICES": "3"},
                                   stdout=handle, stderr=subprocess.STDOUT)
        jobs.append((3, "resolution", "all", process, handle, ["bash", str(worker)], resolution_expected))

    atomic_json(args.output_dir / "run_status.json", {
        "status": "running", "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "jobs": [{"gpu": gpu, "split": split, "kind": kind, "pid": process.pid, "command": cmd}
                 for gpu, split, kind, process, _handle, cmd, _expected in jobs],
        "reused": reused,
        "method_ranking_computed": False, "spent_sealed_split_touched": False,
    })
    failures = []
    for gpu, split, kind, process, handle, cmd, expected in jobs:
        returncode = process.wait()
        handle.close()
        wanted = 512 if split == "conflict" else 128
        bad = [str(path) for path in expected if count_jsonl(path) != wanted]
        if returncode or bad:
            failures.append({"gpu": gpu, "split": split, "kind": kind,
                             "returncode": returncode, "bad_outputs": bad, "command": cmd})
    status = "failed" if failures else "completed"
    atomic_json(args.output_dir / "run_status.json", {
        "status": status, "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "failures": failures, "reused": reused, "method_ranking_computed": False,
        "spent_sealed_split_touched": False,
    })
    if failures:
        raise RuntimeError(json.dumps(failures, indent=2))


if __name__ == "__main__":
    main()
