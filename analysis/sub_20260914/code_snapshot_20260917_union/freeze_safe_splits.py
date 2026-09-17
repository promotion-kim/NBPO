"""Freeze the policy-panel splits for the Safe contract, disjoint from the audit.

The screening panel's 200 prompts are removed first, so no prompt that was used
to decide whether to run the panel can appear in training, development or the
final test set. Selection is again a namespaced hash of the normalized prompt,
so it cannot see a label, and the three splits are drawn in one pass from the
same ordering.

The declared plan is 2,000 train / 500 dev / 1,000 test. A cost-based reduction
to 1,000 train is permitted by the contract only if it is frozen before any
outcome is inspected, so this file writes BOTH the 2,000-prompt ordering and
the 1,000-prompt prefix of it, and records which one the campaign committed to
together with the measured throughput that justified the choice. The prefix
relation means the reduced train set is a subset of the full one rather than a
different draw.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path("/work/sub_20260914")
SNAPSHOT = ("/work/hf_cache/hub/datasets--PKU-Alignment--PKU-SafeRLHF/"
            "snapshots/9421ffafec3fa40a1f1a7d567b4d525079477ecb")
NS_SPLIT = "sub_20260914-safe-split:"
NS_PANEL = "sub_20260914-safe-panel:"
MIN_CHARS, MAX_CHARS = 16, 1200


def digest(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def file_hash(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).replace("​", "")
    return re.sub(r"\s+", " ", t).strip().lower()


def eligible():
    seen, rows = set(), []
    stats = Counter()
    for path in sorted(glob.glob(SNAPSHOT + "/data/*/train.jsonl")):
        source_model = Path(path).parent.name
        with open(path) as stream:
            for line in stream:
                if not line.strip():
                    continue
                d = json.loads(line)
                stats["rows_total"] += 1
                norm = normalize(d.get("prompt") or "")
                if not (MIN_CHARS <= len(norm) <= MAX_CHARS):
                    stats["rejected_length"] += 1
                    continue
                if norm in seen:
                    stats["rejected_duplicate_prompt"] += 1
                    continue
                seen.add(norm)
                rows.append({"prompt_id": digest(NS_PANEL + norm)[:16],
                             "instruction": (d.get("prompt") or "").strip(),
                             "normalized_sha256": digest(norm),
                             "source": "PKU-SafeRLHF/" + source_model,
                             "prompt_source": d.get("prompt_source")})
                stats["eligible_unique"] += 1
    return rows, stats


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        for r in rows:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    return file_hash(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-dev", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-train-reduced", type=int, default=1000)
    args = ap.parse_args()

    out = ROOT / "splits"
    if (out / "freeze.json").exists():
        print(json.dumps({"skipped": "splits already frozen"}))
        return 0

    audit = {json.loads(l)["prompt_id"]
             for l in (ROOT / "panel" / "screen200.jsonl").open() if l.strip()}
    rows, stats = eligible()
    pool = [r for r in rows if r["prompt_id"] not in audit]
    stats["removed_audit_prompts"] = len(rows) - len(pool)

    ordered = sorted(pool, key=lambda r: (digest(NS_SPLIT + r["normalized_sha256"]),
                                          r["prompt_id"]))
    need = args.n_train + args.n_dev + args.n_test
    if len(ordered) < need:
        raise SystemExit("pool of %d is smaller than the %d prompts declared"
                         % (len(ordered), need))
    train = ordered[:args.n_train]
    dev = ordered[args.n_train:args.n_train + args.n_dev]
    test = ordered[args.n_train + args.n_dev:need]
    train_reduced = train[:args.n_train_reduced]

    ids = [{r["prompt_id"] for r in s} for s in (train, dev, test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert not (ids[0] | ids[1] | ids[2]) & audit, "an audit prompt reached a split"

    hashes = {"train2000.jsonl": write_jsonl(out / "train2000.jsonl", train),
              "train1000.jsonl": write_jsonl(out / "train1000.jsonl", train_reduced),
              "dev500.jsonl": write_jsonl(out / "dev500.jsonl", dev),
              "test1000.jsonl": write_jsonl(out / "test1000.jsonl", test)}

    manifest = {
        "experiment_id": "sub_20260914",
        "dataset": "PKU-Alignment/PKU-SafeRLHF",
        "dataset_snapshot": SNAPSHOT.rsplit("/", 1)[-1],
        "objectives": ["helpfulness", "harmlessness"],
        "selection_rule": "sort the audit-free pool by SHA256('%s' + normalized_prompt)" % NS_SPLIT,
        "disjointness": {"audit_prompts_removed_first": True,
                         "train_dev_test_pairwise_disjoint": True,
                         "keyed_on": "NFKC + whitespace-collapsed lowercase prompt"},
        "counts": {"train_full": len(train), "train_reduced": len(train_reduced),
                   "dev": len(dev), "test": len(test), **dict(stats)},
        "reduced_is_a_prefix_of_full": True,
        "committed_train_split": None,
        "commitment_rule": ("set committed_train_split to train2000 or train1000 from the "
                            "measured pilot throughput BEFORE any policy is trained or any "
                            "outcome inspected, and record the throughput that justified it"),
        "sources": {"train": dict(Counter(r["source"] for r in train)),
                    "dev": dict(Counter(r["source"] for r in dev)),
                    "test": dict(Counter(r["source"] for r in test))},
        "split_sha256": hashes,
        "source_sha256": file_hash(__file__)}
    (out / "freeze.json").write_text(json.dumps(manifest, indent=2,
                                                ensure_ascii=False) + "\n")
    print(json.dumps({k: manifest[k] for k in ("counts", "split_sha256")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
