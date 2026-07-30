#!/usr/bin/env python3
"""Run one frozen decode and record its status in W&B."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import wandb


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--repo", required=True)
    args = p.parse_args()

    if args.name == "ronpo_topmass":
        sentinel = Path("/NHNHOME/AIPR/sjkim/novelty_defense_20260723/checkpoints/TRANSFER_COMPLETE")
        deadline = time.time() + 3600
        while not sentinel.exists() and time.time() < deadline:
            time.sleep(10)
        if not sentinel.exists():
            raise TimeoutError("top-mass checkpoint transfer did not complete within one hour")

    config = {
        "model": args.model,
        "data": args.data,
        "seed": 42,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 2048,
        "dtype": "bfloat16",
        "evaluation": "novelty_defense_20260723",
    }
    run = wandb.init(
        entity="promotion-kim",
        project="mnpo",
        job_type="evaluation-decode",
        name=f"novelty-defense-decode-{args.name}",
        config=config,
        tags=["novelty-defense", "joint-eval", "decode"],
    )
    cmd = [
        args.python, "-m", "on_policy_data_gen.decode",
        "--data_dir", args.data,
        "--model", args.model,
        "--temperature", "0.7",
        "--top_p", "0.9",
        "--max_tokens", "2048",
        "--output_dir", args.output,
        "--num_gpu", "1",
        "--batch_size", "64",
        "--seeds", "42",
        "--dtype", "bfloat16",
        "--gpu_memory_utilization", "0.90",
    ]
    started = time.time()
    status = "failed"
    try:
        subprocess.run(cmd, cwd=args.repo, check=True)
        output = Path(args.output) / "output_42.json"
        rows = len(json.loads(output.read_text()))
        if rows != 647:
            raise RuntimeError(f"expected 647 rows, found {rows}")
        run.summary.update({"rows": rows, "elapsed_seconds": time.time() - started})
        status = "finished"
    finally:
        run.summary["decode_status"] = status
        run.finish(exit_code=0 if status == "finished" else 1)


if __name__ == "__main__":
    main()
