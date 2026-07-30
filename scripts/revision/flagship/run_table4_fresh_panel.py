#!/usr/bin/env python3
"""Run the locked two-judge panel once on already-gated fresh generations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def run(command: list[str], *, cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--flagship-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--evaluator-lock", type=Path, required=True)
    parser.add_argument("--metric-lock", type=Path, required=True)
    parser.add_argument("--qwen-judge", type=Path, required=True)
    parser.add_argument("--gpt-judge", type=Path, required=True)
    args = parser.parse_args()
    work = args.run_dir / "fresh_test"
    logs = work / "logs"
    gates_path = work / "stability_gates/summary.json"
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    eligible = sorted(gates["eligible_models"])
    if "base" not in eligible or len(eligible) < 2:
        raise RuntimeError("fresh stability gate has no comparison set")
    generations = work / "generations"
    judge_input = work / "judge/input_tasks.jsonl"
    judge_lock = work / "judge/input_lock.json"
    prepare = [
        args.python, str(args.project / "scripts/revision/flagship/prepare_fair_demo_judge_inputs.py"),
        "--split", "fresh_test", "--base", str(generations / "base/output_42.json"),
        "--evaluator-lock", str(args.evaluator_lock), "--output", str(judge_input),
        "--lock-output", str(judge_lock),
    ]
    for candidate in eligible:
        if candidate != "base":
            prepare.extend(["--candidate", f"{candidate}={generations / candidate / 'output_42.json'}"])
    run(prepare, cwd=args.project, log=logs / "prepare_table4_judge_input.log")
    jobs = []
    specs = [
        ("qwen3_32b", args.qwen_judge, 0, 0), ("qwen3_32b", args.qwen_judge, 1, 1),
        ("gpt_oss_120b", args.gpt_judge, 2, 0), ("gpt_oss_120b", args.gpt_judge, 3, 1),
    ]
    for judge, model, gpu, shard in specs:
        output = work / f"judge/{judge}/shard_{shard}.jsonl"
        command = [
            args.python, str(args.project / "scripts/revision/flagship/judge_fair_demo_models.py"),
            "--input", str(judge_input), "--input-lock", str(judge_lock),
            "--evaluator-lock", str(args.evaluator_lock), "--judge-id", judge,
            "--model-path", str(model), "--output", str(output),
            "--shard-index", str(shard), "--num-shards", "2", "--batch-size", "128",
            "--max-model-len", "16384",
        ]
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(args.project),
            "HF_HOME": str(args.flagship_root / "cache/huggingface"),
            "HF_HUB_CACHE": str(args.flagship_root / "cache/huggingface/hub"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "TOKENIZERS_PARALLELISM": "false",
        })
        handle = (logs / f"table4_judge_{judge}_shard{shard}.log").open("a", encoding="utf-8")
        process = subprocess.Popen(command, cwd=args.project, env=environment,
                                   stdout=handle, stderr=subprocess.STDOUT)
        jobs.append((judge, shard, process, handle, command))
    status_path = work / "panel_status.json"
    atomic_json(status_path, {
        "status": "running", "stage": "locked_fresh_panel", "eligible_models": eligible,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "spent_sealed_split_touched": False,
    })
    failures = []
    for judge, shard, process, handle, command in jobs:
        returncode = process.wait()
        handle.close()
        if returncode:
            failures.append({"judge": judge, "shard": shard, "returncode": returncode, "command": command})
    if failures:
        atomic_json(status_path, {"status": "failed", "stage": "locked_fresh_panel",
                                  "failures": failures, "spent_sealed_split_touched": False})
        raise RuntimeError(json.dumps(failures, indent=2))
    old_output = work / "results/panel_legacy"
    aggregate = [
        args.python, str(args.project / "scripts/revision/flagship/aggregate_fair_demo_panel.py"),
        "--input", str(judge_input), "--input-lock", str(judge_lock),
        "--evaluator-lock", str(args.evaluator_lock),
        "--judge-dir", f"qwen3_32b={work / 'judge/qwen3_32b'}",
        "--judge-dir", f"gpt_oss_120b={work / 'judge/gpt_oss_120b'}",
        "--gates", str(gates_path), "--output-dir", str(old_output),
    ]
    run(aggregate, cwd=args.project, log=logs / "aggregate_table4_panel_legacy.log")
    result_output = args.run_dir / "fresh"
    marginal = [
        args.python, str(args.project / "analysis/ronpo_8b_reconstruction_20260714/build_worstobj_table.py"),
        "--prompt-scores", str(old_output / "panel_prompt_scores.jsonl"),
        "--metric-lock", str(args.metric_lock), "--gates", str(gates_path),
        "--grid", str(args.grid), "--output-dir", str(result_output), "--split", "fresh_1024",
    ]
    run(marginal, cwd=args.project, log=logs / "aggregate_table4_marginal.log")
    atomic_json(status_path, {
        "status": "completed", "stage": "locked_fresh_panel_aggregated",
        "eligible_models": eligible, "result": str(result_output / "panel_summary.json"),
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "spent_sealed_split_touched": False,
    })


if __name__ == "__main__":
    main()
