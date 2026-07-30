#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import time
from pathlib import Path


OBJECTIVES = ("instruction_following", "truthfulness", "honesty", "helpfulness", "safety")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_stable_jsonl(path, attempts=8):
    """Read an immutable NFS shard only after two identical valid reads."""
    previous = None
    error = None
    for _ in range(attempts):
        try:
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            rows = [json.loads(line) for line in data.splitlines() if line]
            current = (len(data), digest)
            if current == previous:
                return rows, digest
            previous = current
        except Exception as exc:
            error = exc
            previous = None
        time.sleep(1)
    raise RuntimeError(f"unstable or invalid shard {path}: {error}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--expected-count", type=int, required=True)
    p.add_argument("--audit", type=Path, required=True)
    a = p.parse_args()
    audit = {"expected_count": a.expected_count, "objectives": {}}
    prompt_order = None
    for objective in OBJECTIVES:
        rows, shards = [], []
        for shard in range(8):
            path = a.scores / f"joint_{objective}.jsonl.shard{shard}"
            shard_rows, digest = read_stable_jsonl(path)
            rows.extend(shard_rows)
            shards.append({"shard": shard, "rows": len(shard_rows), "sha256": digest})
        if len(rows) != a.expected_count:
            raise RuntimeError(f"{objective}: expected {a.expected_count} rows, got {len(rows)}")
        prompts = [row["prompt"] for row in rows]
        if len(set(prompts)) != len(prompts):
            raise RuntimeError(f"{objective}: duplicate prompts")
        if prompt_order is None:
            prompt_order = prompts
        elif prompts != prompt_order:
            raise RuntimeError(f"{objective}: prompt order mismatch")
        out = a.scores / f"joint_{objective}.jsonl"
        tmp = out.with_suffix(out.suffix + ".tmp")
        with tmp.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(out)
        audit["objectives"][objective] = {
            "rows": len(rows), "shards": shards, "combined_sha256": sha256(out)
        }
    a.audit.write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
