"""Queue tonight's bank-diagnostic judging: one gating pilot, then four shards.

Priority 51/52 puts these ahead of the not-yet-started PROSPER seed 44 (70) and
the six DPO weight arms (71-76), which is the instruction for tonight, and
behind the running PROSPER seed 43 chain (44/45/67) so a healthy DDP run and its
evaluation finish undisturbed.

The pilot gates the shards: ten prompts measure cold load, tokens/s, parse rate
and complete-prompt rate on real verdicts before 190 more prompts are committed,
and its verdicts belong to the panel rather than being discarded.
"""
import json
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
QUEUE = ROOT / "jobs" / "queue"
OUT = ROOT / "analysis/diag_20260914/bank4"
ENV = {"PYTHONPATH": "/work/pylibs_eval:/work/uf4_20260910/code",
       "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
       "TOKENIZERS_PARALLELISM": "false", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
       "OMP_NUM_THREADS": "4", "WANDB_MODE": "disabled"}


def write(spec):
    existing = {json.loads(p.read_text())["job_id"] for p in QUEUE.glob("*.json")}
    if spec["job_id"] in existing:
        return "exists  " + spec["job_id"]
    path = QUEUE / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(spec, indent=2) + "\n")
    tmp.replace(path)
    return "queued  " + spec["job_id"]


def main():
    base = ["python3", str(ROOT / "code/diag_judge_bank.py")]
    print(write({
        "job_id": "uf4_diag_bank_pilot10", "priority": 51, "gpus": 1,
        "cwd": str(ROOT), "env": ENV, "depends_on": [],
        "command": base + ["--first", "10"],
        "timeout_s": 7200,
        "artifacts": [str(OUT / "complete_first10.json")],
    }))
    for shard in range(4):
        print(write({
            "job_id": "uf4_diag_bank_shard%d" % shard, "priority": 52, "gpus": 1,
            "cwd": str(ROOT), "env": ENV,
            "depends_on": ["uf4_diag_bank_pilot10"],
            "command": base + ["--shard", str(shard), "--nshards", "4"],
            "timeout_s": 14400,
            "artifacts": [str(OUT / ("complete_shard%dof4.json" % shard))],
        }))


if __name__ == "__main__":
    main()
