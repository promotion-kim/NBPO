"""Write splits/safe_v1 in the UF pool format and queue the 8Y+8Z generation.

The frozen Safe splits already carry prompt_id, instruction and source; the pool
generator needs exactly those three fields, so the conversion is a projection
rather than a new selection. Nothing is re-sorted or re-sampled here: the row
order is the frozen order, and the sha256 of each frozen file is recorded
alongside the converted one.
"""
import json
from pathlib import Path

SUB = Path("/work/sub_20260914")
UF = Path("/work/uf4_20260910")
OUT = UF / "splits/safe_v1"
QUEUE = UF / "jobs/queue"
CODE = "/work/uf4_20260910/code"
MAP = {"policy_train": "train2000.jsonl", "policy_dev": "dev500.jsonl",
       "final_eval": "test1000.jsonl"}
ENV = {"PYTHONPATH": "/work/pylibs_eval:/work/uf4_20260910/code",
       "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
       "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "OMP_NUM_THREADS": "4",
       "WANDB_MODE": "disabled"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    written = {}
    for split, source in MAP.items():
        dest = OUT / ("%s.jsonl" % split)
        if dest.exists():
            written[split] = "exists"
            continue
        rows = [json.loads(l) for l in (SUB / "splits" / source).open() if l.strip()]
        with dest.open("x") as stream:
            for r in rows:
                stream.write(json.dumps({"prompt_id": r["prompt_id"],
                                         "source": r["source"],
                                         "instruction": r["instruction"],
                                         "normalized_sha256": r["normalized_sha256"]},
                                        ensure_ascii=False) + "\n")
        written[split] = len(rows)
    print(json.dumps({"splits_written": written}))

    existing = {json.loads(p.read_text())["job_id"] for p in QUEUE.glob("*.json")}
    specs = []
    for split, out_name, shards, prio in (("policy_train", "safe_v1", 4, 69),
                                          ("policy_dev", "safe_dev_v1", 2, 69)):
        for shard in range(shards):
            specs.append({
                "job_id": "sub_pool_%s_shard%d" % (out_name, shard),
                "priority": prio, "gpus": 1, "depends_on": [],
                "command": ["python3", CODE + "/generate_uf4_pool.py",
                            "--splits", "safe_v1", "--split-names", split,
                            "--out-name", out_name,
                            "--shard", str(shard), "--shards", str(shards)],
                "artifacts": ["/work/uf4_20260910/pools/%s/shard%d/settings.json"
                              % (out_name, shard)]})
    for spec in specs:
        spec = {**spec, "cwd": CODE, "env": ENV, "timeout_s": 43200}
        if spec["job_id"] in existing:
            print("exists  " + spec["job_id"]); continue
        path = QUEUE / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(spec, indent=2) + "\n")
        tmp.replace(path)
        print("queued  p%-3d %s" % (spec["priority"], spec["job_id"]))


if __name__ == "__main__":
    main()
