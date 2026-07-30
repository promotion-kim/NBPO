#!/usr/bin/env python3
"""Score the locked pool with all preregistered objectives on four GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_output(source: Path, root: Path, objective: str) -> dict:
    handles = {}
    paths = {}
    counts = {}
    try:
        for split in ["general", "conflict_curated"]:
            path = root / "scores" / split / f"{objective}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            handles[split] = path.open("w", encoding="utf-8")
            paths[split] = path
            counts[split] = 0
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                split = str(row["split"])
                handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    if counts != {"general": 647, "conflict_curated": 128}:
        raise RuntimeError(f"unexpected split counts for {objective}: {counts}")
    return {split: {"path": str(paths[split]), "sha256": sha(paths[split]), "rows": counts[split]} for split in paths}


def task_command(task: dict, repo: Path, root: Path, python: str) -> list[str]:
    inp = root / "pools/combined.jsonl"
    out = root / "scores_raw" / f"{task['objective']}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    kind = task["kind"]
    if kind == "skywork":
        return [python, str(repo / "on_policy_data_gen/rm_skywork.py"), "--model_name", task["model"], "--cache_dir", str(Path(task["model"]).parents[2]), "--input_file", str(inp), "--output_file", str(out), "--local_files_only", "--attn_implementation", "eager", "--batch_size", "16", "--sample_batch_size", "8", "--max_seq_length", "4096"]
    if kind == "athene":
        return [python, str(repo / "on_policy_data_gen/rm_athene.py"), "--model_name", task["model"], "--cache_dir", str(Path(task["model"]).parents[2]), "--input_file", str(inp), "--output_file", str(out), "--local_files_only", "--batch_size", "16", "--sample_batch_size", "8"]
    if kind == "armo":
        command = [python, str(repo / "on_policy_data_gen/rm_armo.py"), "--cache_dir", str(Path(task["model"]).parents[2]), "--input_file", str(inp), "--output_file", str(out), "--device", "cuda", "--batch_size", "16", "--sample_batch_size", "8", "--max_seq_length", "4096", "--local_files_only", "True", "--revision", "eb2676d20da2f2d41082289d23c59b9f7427f955"]
        if task.get("attribute"):
            command.extend(["--reward_attribute_name", task["attribute"]])
        return command
    if kind == "guard":
        return [python, str(repo / "on_policy_data_gen/rm_qwen3guard.py"), "--model_name", task["model"], "--cache_dir", str(Path(task["model"]).parents[2]), "--input_file", str(inp), "--output_file", str(out), "--device", "cuda", "--batch_size", "32", "--sample_batch_size", "8", "--max_input_length", "4096", "--max_new_tokens", "48", "--local_files_only"]
    raise ValueError(kind)


def run_task(task: dict, gpu: int, repo: Path, root: Path, python: str) -> dict:
    objective = task["objective"]
    log = root / "logs" / f"score_{objective}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = task_command(task, repo, root, python)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TORCH_CUDNN_SDPA_ENABLED"] = "0"
    with log.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND=" + json.dumps(command) + "\n")
        subprocess.run(command, cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    split = split_output(root / "scores_raw" / f"{objective}.jsonl", root, objective)
    return {"objective": objective, "gpu": gpu, "status": "completed", "outputs": split}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--skywork", required=True)
    parser.add_argument("--athene", required=True)
    parser.add_argument("--armo", required=True)
    parser.add_argument("--guard", required=True)
    args = parser.parse_args()
    tasks = [
        {"objective": "skywork_v2", "kind": "skywork", "model": args.skywork},
        {"objective": "athene", "kind": "athene", "model": args.athene},
        {"objective": "armo_whole", "kind": "armo", "model": args.armo},
        {"objective": "armo_helpfulness", "kind": "armo", "model": args.armo, "attribute": "ultrafeedback-helpfulness"},
        {"objective": "armo_safety", "kind": "armo", "model": args.armo, "attribute": "beavertails-is_safe"},
        {"objective": "armo_conciseness", "kind": "armo", "model": args.armo, "attribute": "helpsteer-verbosity"},
        {"objective": "qwen3guard_safety", "kind": "guard", "model": args.guard},
    ]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending = iter(tasks)
        active = {}
        for gpu in range(4):
            task = next(pending, None)
            if task:
                active[pool.submit(run_task, task, gpu, args.repo, args.root, args.python)] = gpu
        while active:
            done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                gpu = active.pop(future)
                results.append(future.result())
                task = next(pending, None)
                if task:
                    active[pool.submit(run_task, task, gpu, args.repo, args.root, args.python)] = gpu
                (args.root / "score_status.json").write_text(json.dumps({"completed": results, "total": len(tasks)}, indent=2) + "\n", encoding="utf-8")

    brevity_raw = args.root / "scores_raw/brevity.jsonl"
    collapse = args.root / "scores_raw/brevity_collapse.csv"
    subprocess.run([
        args.python, str(args.repo / "mnpo_scripts/score_brevity_and_collapse.py"),
        "--input_file", str(args.root / "pools/combined.jsonl"), "--output_file", str(brevity_raw),
        "--collapse_csv", str(collapse), "--target_words", "180", "--tolerance_words", "80",
    ], cwd=args.repo, check=True)
    results.append({"objective": "brevity", "gpu": None, "status": "completed", "outputs": split_output(brevity_raw, args.root, "brevity")})
    (args.root / "score_status.json").write_text(json.dumps({"completed": results, "total": 8}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"completed": len(results), "total": 8}, indent=2))


if __name__ == "__main__":
    main()
