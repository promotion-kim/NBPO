#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

OBJECTIVES = ("instruction_following", "truthfulness", "honesty", "helpfulness", "safety")


def read_stable(path, attempts=8):
    previous = None
    for _ in range(attempts):
        data = path.read_bytes()
        state = (len(data), hashlib.sha256(data).hexdigest())
        rows = [json.loads(x) for x in data.splitlines() if x]
        if state == previous:
            return rows, state[1]
        previous = state
        time.sleep(1)
    raise RuntimeError(f"unstable shard {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--num-shards", type=int, default=6)
    p.add_argument("--expected-count", type=int, required=True)
    p.add_argument("--audit", type=Path, required=True)
    a = p.parse_args()
    audit = {"expected_count": a.expected_count, "num_shards": a.num_shards, "objectives": {}}
    order = None
    for objective in OBJECTIVES:
        rows, shards = [], []
        for shard in range(a.num_shards):
            path = a.scores / f"joint_{objective}.jsonl.shard{shard}"
            part, digest = read_stable(path)
            rows.extend(part); shards.append({"shard": shard, "rows": len(part), "sha256": digest})
        if len(rows) != a.expected_count:
            raise RuntimeError(f"{objective}: {len(rows)} != {a.expected_count}")
        prompts = [x["prompt"] for x in rows]
        if len(set(prompts)) != len(prompts):
            raise RuntimeError(f"duplicate {objective} prompts")
        if order is None: order = prompts
        elif prompts != order: raise RuntimeError("objective prompt order mismatch")
        out = a.scores / f"joint_{objective}.jsonl"; tmp = out.with_suffix(".jsonl.tmp")
        with tmp.open("w") as f:
            for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush(); os.fsync(f.fileno())
        tmp.replace(out)
        audit["objectives"][objective] = {"rows": len(rows), "shards": shards, "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    a.audit.write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
