"""Queue the Safe screening chain on the existing controller.

The controller in /work/uf4_20260910 is the only scheduler and is not
duplicated: these specs simply join its queue with their own cwd and
environment. The chain is candidates -> pilot judging -> the remaining 190
prompts, with the pilot first because the contract requires a measured
throughput and parse rate before the full screening budget is committed.

Priorities sit above the UF weight arms, which were demoted to 88-91, and
below nothing that is currently running: the s44 final-eval chain at 62/63
keeps its cards.
"""
import json
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
QUEUE = ROOT / "jobs/queue"
CODE = "/work/sub_20260914/code"
ENV = {"PYTHONPATH": "/work/pylibs_eval:/work/sub_20260914/code",
       "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
       "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "OMP_NUM_THREADS": "4",
       "WANDB_MODE": "disabled"}

SPECS = [
    {"job_id": "sub_gen_safe_screen200", "priority": 64, "gpus": 1,
     "depends_on": [],
     "command": ["python3", CODE + "/gen_candidates.py", "--panel", "screen200.jsonl"],
     "artifacts": ["/work/sub_20260914/responses/screen200/complete.json"]},
    {"job_id": "sub_judge_safe_pilot10", "priority": 65, "gpus": 1,
     "depends_on": ["sub_gen_safe_screen200"],
     "command": ["python3", CODE + "/judge_pairs.py", "--responses", "screen200",
                 "--prompt-list", "pilot10.jsonl", "--out-tag", "safe_pilot10_panelA",
                 "--panel", "A", "--draws-per-order", "2"],
     "artifacts": ["/work/sub_20260914/judgments/safe_pilot10_panelA/complete.json"]},
]


def main():
    existing = {json.loads(p.read_text())["job_id"] for p in QUEUE.glob("*.json")}
    for spec in SPECS:
        spec = {**spec, "cwd": CODE, "env": ENV, "timeout_s": 43200}
        if spec["job_id"] in existing:
            print("exists  " + spec["job_id"])
            continue
        path = QUEUE / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(spec, indent=2) + "\n")
        tmp.replace(path)
        print("queued  p%-3d %s" % (spec["priority"], spec["job_id"]))


if __name__ == "__main__":
    main()
