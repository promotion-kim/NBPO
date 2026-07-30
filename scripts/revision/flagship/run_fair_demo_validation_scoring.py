#!/usr/bin/env python3
"""Score gated validation candidates with the locked RM triple and independent judge panel."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(bool(line.strip()) for line in path.open(encoding="utf-8"))


def run(command: list[str], *, cwd: Path, log: Path, env: dict[str, str] | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--flagship-root", type=Path, required=True)
    parser.add_argument("--fair-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "fresh_test"], default="validation")
    parser.add_argument("--evaluator-lock", type=Path, required=True)
    parser.add_argument("--general-rm-cache", type=Path, required=True)
    parser.add_argument("--qwen-judge", type=Path, required=True)
    parser.add_argument("--gpt-judge", type=Path, required=True)
    args = parser.parse_args()
    evaluator = json.loads(args.evaluator_lock.read_text(encoding="utf-8"))
    if evaluator.get("status") != "LOCKED_BEFORE_ANY_NEW_METHOD_RANKING":
        raise RuntimeError("evaluator must be locked before validation scoring")
    work = args.run_dir / args.split
    logs = work / "logs"
    status_path = work / "scoring_status.json"
    gates_path = work / "stability_gates/summary.json"
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    eligible = gates["eligible_models"]
    if "base" not in eligible or len(eligible) < 2:
        raise RuntimeError("no eligible validation candidate")
    generations = work / "generations"
    merged = work / "merged_generations.json"
    merge = [args.python, "-m", "mnpo_scripts.merge_model_generations", "--generations"]
    merge.extend([f"{model}={generations / model / 'output_42.json'}" for model in eligible])
    merge.extend(["--output_file", str(merged)])
    run(merge, cwd=args.project, log=logs / "merge.log")
    prompt_count = len(json.loads(merged.read_text(encoding="utf-8")))

    signals = evaluator["objective_signals"]
    scores = work / "scores"
    scores.mkdir(parents=True, exist_ok=True)
    jobs = []
    armo_needed = any(signal.startswith("armo_") for signal in signals)
    general_signal = "skywork" if "skywork" in signals else ("athene" if "athene" in signals else None)
    armo_outputs = [scores / "armo/helpfulness.jsonl", scores / "armo/safety.jsonl",
                    scores / "armo/conciseness.jsonl"]
    armo_complete = all(count_jsonl(path) == prompt_count for path in armo_outputs)
    if armo_needed and not armo_complete:
        armo = [
            args.python, str(args.project / "scripts/revision/flagship/score_armo_primary_heads_sharded.py"),
            "--python", args.python, "--input-file", str(merged), "--output-dir", str(scores / "armo"),
            "--cache-dir", str(args.flagship_root / "cache/huggingface/hub"),
            "--gpu-ids", *("0 1 2".split() if general_signal else "0 1 2 3".split()),
            "--batch-size", "8", "--sample-batch-size", "4", "--local-files-only",
        ]
        environment = os.environ.copy()
        environment.update({"PYTHONPATH": str(args.project), "HF_HUB_OFFLINE": "1",
                            "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                            "TORCH_CUDNN_SDPA_ENABLED": "0"})
        handle = (logs / "score_armo.log").open("a", encoding="utf-8")
        process = subprocess.Popen(armo, cwd=args.project, env=environment,
                                   stdout=handle, stderr=subprocess.STDOUT)
        jobs.append(("armo", process, handle, armo))
    general_output = scores / f"{general_signal}.jsonl" if general_signal else None
    general_complete = bool(general_output is not None and count_jsonl(general_output) == prompt_count)
    if general_signal and not general_complete:
        model = ("Skywork/Skywork-Reward-V2-Llama-3.1-8B" if general_signal == "skywork"
                 else "Nexusflow/Athene-RM-8B")
        revision = ("cba2f842f3f1af2f1b2f0d35e794d789976390c5" if general_signal == "skywork"
                    else "cdf428f7b52a323b6cf4e9803e5bcba9f1fb5a59")
        command = [
            args.python, "-u", "-m", f"on_policy_data_gen.rm_{general_signal}",
            "--input_file", str(merged), "--output_file", str(scores / f"{general_signal}.jsonl"),
            "--cache_dir", str(args.general_rm_cache), "--model_name", model, "--revision", revision,
            "--local_files_only", "--batch_size", "16", "--sample_batch_size", "8",
        ]
        if general_signal == "skywork":
            command.extend(["--max_seq_length", "4096", "--attn_implementation", "sdpa"])
        environment = os.environ.copy()
        environment.update({"CUDA_VISIBLE_DEVICES": "3", "PYTHONPATH": str(args.project),
                            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                            "TOKENIZERS_PARALLELISM": "false", "TORCH_CUDNN_SDPA_ENABLED": "0"})
        handle = (logs / f"score_{general_signal}.log").open("a", encoding="utf-8")
        process = subprocess.Popen(command, cwd=args.project, env=environment,
                                   stdout=handle, stderr=subprocess.STDOUT)
        jobs.append((general_signal, process, handle, command))
    atomic_json(status_path, {"status": "running", "stage": "locked_reward_scoring",
                              "signals": signals, "eligible_models": eligible,
                              "spent_sealed_split_touched": False,
                              "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")})
    failures = []
    for name, process, handle, command in jobs:
        returncode = process.wait(); handle.close()
        if returncode:
            failures.append({"name": name, "returncode": returncode, "command": command})
    if failures:
        atomic_json(status_path, {"status": "failed", "stage": "locked_reward_scoring",
                                  "failures": failures, "spent_sealed_split_touched": False})
        raise RuntimeError(json.dumps(failures, indent=2))
    score_map = {
        "skywork": scores / "skywork.jsonl", "athene": scores / "athene.jsonl",
        "armo_helpfulness": scores / "armo/helpfulness.jsonl",
        "armo_safety": scores / "armo/safety.jsonl",
        "armo_conciseness": scores / "armo/conciseness.jsonl",
    }
    for signal in signals:
        if signal != "length_conciseness" and count_jsonl(score_map[signal]) != prompt_count:
            raise RuntimeError(f"incomplete locked reward output for {signal}")
    reward_command = [
        args.python, str(args.project / "scripts/revision/flagship/aggregate_fair_demo_rewards.py"),
        "--merged", str(merged), "--evaluator-lock", str(args.evaluator_lock),
        "--output-dir", str(work / "results/rewards"),
    ]
    for signal in signals:
        if signal != "length_conciseness":
            reward_command.extend(["--score", f"{signal}={score_map[signal]}"])
    run(reward_command, cwd=args.project, log=logs / "aggregate_rewards.log")

    judge_input = work / "judge/input_tasks.jsonl"
    judge_lock = work / "judge/input_lock.json"
    prepare = [
        args.python, str(args.project / "scripts/revision/flagship/prepare_fair_demo_judge_inputs.py"),
        "--split", args.split, "--base", str(generations / "base/output_42.json"),
        "--evaluator-lock", str(args.evaluator_lock), "--output", str(judge_input),
        "--lock-output", str(judge_lock),
    ]
    for candidate in eligible:
        if candidate != "base":
            prepare.extend(["--candidate", f"{candidate}={generations / candidate / 'output_42.json'}"])
    run(prepare, cwd=args.project, log=logs / "prepare_judge_input.log")
    judge_jobs = []
    judge_specs = [
        ("qwen3_32b", args.qwen_judge, 0, 0), ("qwen3_32b", args.qwen_judge, 1, 1),
        ("gpt_oss_120b", args.gpt_judge, 2, 0), ("gpt_oss_120b", args.gpt_judge, 3, 1),
    ]
    for judge, model, gpu, shard in judge_specs:
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
        environment.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(args.project),
                            "HF_HOME": str(args.flagship_root / "cache/huggingface"),
                            "HF_HUB_CACHE": str(args.flagship_root / "cache/huggingface/hub"),
                            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                            "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "TOKENIZERS_PARALLELISM": "false"})
        handle = (logs / f"judge_{judge}_shard{shard}.log").open("a", encoding="utf-8")
        process = subprocess.Popen(command, cwd=args.project, env=environment,
                                   stdout=handle, stderr=subprocess.STDOUT)
        judge_jobs.append((judge, shard, process, handle, command))
    atomic_json(status_path, {"status": "running", "stage": "locked_validation_judge_panel",
                              "signals": signals, "eligible_models": eligible,
                              "spent_sealed_split_touched": False,
                              "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")})
    failures = []
    for judge, shard, process, handle, command in judge_jobs:
        returncode = process.wait(); handle.close()
        if returncode:
            failures.append({"judge": judge, "shard": shard, "returncode": returncode, "command": command})
    if failures:
        atomic_json(status_path, {"status": "failed", "stage": "locked_validation_judge_panel",
                                  "failures": failures, "spent_sealed_split_touched": False})
        raise RuntimeError(json.dumps(failures, indent=2))
    aggregate = [
        args.python, str(args.project / "scripts/revision/flagship/aggregate_fair_demo_panel.py"),
        "--input", str(judge_input), "--input-lock", str(judge_lock),
        "--evaluator-lock", str(args.evaluator_lock),
        "--judge-dir", f"qwen3_32b={work / 'judge/qwen3_32b'}",
        "--judge-dir", f"gpt_oss_120b={work / 'judge/gpt_oss_120b'}",
        "--gates", str(gates_path), "--output-dir", str(work / "results/panel"),
    ]
    if args.split == "validation":
        aggregate.extend(["--grid", str(args.grid), "--select"])
    run(aggregate, cwd=args.project, log=logs / "aggregate_panel.log")
    atomic_json(status_path, {"status": "completed",
                              "stage": ("validation_selection_locked" if args.split == "validation"
                                        else "fresh_test_scoring_completed"),
                              "selection_lock": (str(work / "results/panel/selection_lock.json")
                                                 if args.split == "validation" else None),
                              "spent_sealed_split_touched": False,
                              "completed_at": datetime.now().astimezone().isoformat(timespec="seconds")})


if __name__ == "__main__":
    main()
