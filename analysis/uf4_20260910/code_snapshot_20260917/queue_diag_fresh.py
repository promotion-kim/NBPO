"""Queue the fresh-response rows: generate one draw per panel prompt, then judge it.

Three frozen seed-42 checkpoints (NBPO, utilitarian, uniform scalarized DPO) at
priority 53/54, so they follow the bank shards and still precede the
not-yet-started PROSPER seed 44. A fresh row is 200 generated responses and
6,400 rubric-order verdicts; the optional base draw is deliberately not queued
here, since it is a separate declared row.
"""
import json
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
QUEUE = ROOT / "jobs" / "queue"
DIAG = ROOT / "analysis/diag_20260914"
ARMS = [("nbpo_mse_s42", "NBPO"), ("util_mse_s42", "utilitarian"),
        ("dpo_uniform_mse_s42", "uniform scalarized DPO")]
GEN_ENV = {"PYTHONPATH": "/work/pylibs_eval:/work/uf4_20260910/code",
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
    for arm, _label in ARMS:
        gen = "uf4_diag_fresh_gen_%s" % arm
        print(write({
            "job_id": gen, "priority": 53, "gpus": 1, "cwd": str(ROOT), "env": GEN_ENV,
            "depends_on": [],
            "command": ["python3", str(ROOT / "code/diag_gen_fresh.py"),
                        "--arm", arm, "--model", str(ROOT / "arms" / arm)],
            "timeout_s": 5400,
            "artifacts": [str(DIAG / "fresh" / ("complete_%s.json" % arm))],
        }))
        print(write({
            "job_id": "uf4_diag_fresh_judge_%s" % arm, "priority": 54, "gpus": 1,
            "cwd": str(ROOT), "env": GEN_ENV, "depends_on": [gen],
            "command": ["python3", str(ROOT / "code/diag_judge_fresh.py"), "--arm", arm],
            "timeout_s": 10800,
            "artifacts": [str(DIAG / "fresh_verdicts" / ("complete_%s.json" % arm))],
        }))


if __name__ == "__main__":
    main()
