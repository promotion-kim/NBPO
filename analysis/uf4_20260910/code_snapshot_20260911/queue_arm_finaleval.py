"""Queue the final-eval pipeline (4 generation shards + one judge) for one policy arm.

Every arm in the main body needs the identical evaluation contract: fresh
responses on the planned 2,000 final prompts, then the frozen independent judge
in both presentation orders. That contract already exists in
generate_uf4_pool.py and judge_final_eval.py; this script only registers it, so
an arm cannot silently be evaluated under a different protocol than its peers.

It refuses to overwrite an existing spec with different bytes, so re-running it
is safe and never rewrites a job the controller has already acted on.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
QUEUE = ROOT / "jobs" / "queue"

GEN_ENV = {"PYTHONPATH": "/work/pylibs_eval:/work/uf4_20260910/code",
           "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
           "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "OMP_NUM_THREADS": "4",
           "WANDB_MODE": "disabled"}


def write(spec, filename):
    path = QUEUE / filename
    payload = json.dumps(spec, indent=2) + "\n"
    if path.exists() and path.read_text() != payload:
        raise SystemExit(f"refusing to change an existing spec: {path}")
    if path.exists():
        print(json.dumps({"unchanged": spec["job_id"]}), flush=True)
        return
    path.write_text(payload)
    print(json.dumps({"queued": spec["job_id"], "gpus": spec["gpus"],
                      "depends_on": spec["depends_on"]}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--depends-on", default=None,
                    help="training job id; omitted when the checkpoint is already on disk")
    ap.add_argument("--gen-priority", type=int, required=True)
    ap.add_argument("--judge-priority", type=int, required=True)
    ap.add_argument("--shards", type=int, default=4)
    args = ap.parse_args()

    arm = args.arm
    model = ROOT / "arms" / arm
    if args.depends_on is None and not (model / "config.json").exists():
        raise SystemExit(f"{model} has no checkpoint and no training dependency was given")
    train_dep = [args.depends_on] if args.depends_on else []

    gen_ids = []
    for shard in range(args.shards):
        job_id = f"uf4_finaleval_gen_{arm}_shard{shard}"
        gen_ids.append(job_id)
        write({"job_id": job_id, "priority": args.gen_priority, "gpus": 1,
               "cwd": str(ROOT), "env": GEN_ENV, "depends_on": train_dep,
               "command": ["python3", str(ROOT / "code/generate_uf4_pool.py"),
                           "--splits", "v1", "--split-names", "final_eval",
                           "--out-name", f"final_{arm}", "--model", str(model),
                           "--shard", str(shard), "--shards", str(args.shards),
                           "--prompts-per-chunk", "25"],
               "timeout_s": 21600,
               "artifacts": [str(ROOT / f"pools/final_{arm}/shard{shard}/complete_shard{shard}.json")]},
              f"{args.gen_priority}_finaleval_{arm}_shard{shard}.json")

    write({"job_id": f"uf4_finaleval_judge_{arm}", "priority": args.judge_priority,
           "gpus": 1, "cwd": str(ROOT), "env": GEN_ENV, "depends_on": gen_ids,
           "command": ["python3", str(ROOT / "code/judge_final_eval.py"),
                       "--arm", arm, "--policy-pool", str(ROOT / f"pools/final_{arm}")],
           "timeout_s": 43200,
           "artifacts": [str(ROOT / f"evaluation/final_eval/{arm}/complete.json")]},
          f"{args.judge_priority}_finaleval_judge_{arm}.json")


if __name__ == "__main__":
    main()
