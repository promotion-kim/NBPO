"""Freeze the UF-4 prompt groups and the five disjoint splits.

Everything here is decided before any model sees a UF-4 prompt: the source
whitelist, the normalization, the duplicate rule, the decontamination bank, the
eligibility rule and the split sizes. Nothing in this file looks at a policy
result, and the fallback for a short pool is written down before the count is
known rather than chosen after seeing it.

Panel objectives are instruction_following, truthfulness, honesty, helpfulness.
`honesty` is a UltraFeedback criterion in its own right and is never renamed to
safety.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/work/uf4_20260910")
SNAPSHOT = Path("/work/hf_cache/hub/datasets--openbmb--UltraFeedback/snapshots/"
                "40b436560ca83a8dba36114c22ab3c66e43f6d5e")
DATASET_REVISION = "40b436560ca83a8dba36114c22ab3c66e43f6d5e"
SOURCES = ("ultrachat", "sharegpt", "evol_instruct")
OBJECTIVES = ("instruction_following", "truthfulness", "honesty", "helpfulness")
SPLIT_SEED = 20260910
SHINGLE = 5
NUM_PERM = 128
BANDS, ROWS_PER_BAND = 16, 8
NEAR_DUP_JACCARD = 0.90
TARGET = {"pm_train": 20000, "pm_dev": 2000, "policy_train": 10000,
          "policy_dev": 1000, "final_eval": 2000}
FALLBACK = {"pm_train": 20000, "pm_dev": 2000, "policy_train": 5000,
            "policy_dev": 1000, "final_eval": 2000}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shingles(norm: str) -> frozenset:
    words = norm.split()
    if len(words) <= SHINGLE:
        return frozenset({" ".join(words)}) if words else frozenset()
    return frozenset(" ".join(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1))


def shingle_ids(sets):
    """Map every distinct shingle to a uint64 id once, deterministically."""
    table = {}
    out = []
    for group in sets:
        ids = np.empty(len(group), dtype=np.uint64)
        for index, item in enumerate(sorted(group)):
            got = table.get(item)
            if got is None:
                got = int.from_bytes(hashlib.blake2b(item.encode("utf-8"), digest_size=8).digest(),
                                     "big")
                table[item] = got
            ids[index] = got
        out.append(ids)
    return out


def minhash(id_arrays, seed=SPLIT_SEED):
    """(N, NUM_PERM) uint64 signature matrix under one fixed permutation family."""
    rng = np.random.default_rng(seed)
    prime = np.uint64((1 << 61) - 1)
    a = rng.integers(1, int(prime) - 1, size=NUM_PERM, dtype=np.uint64)
    b = rng.integers(0, int(prime) - 1, size=NUM_PERM, dtype=np.uint64)
    signatures = np.full((len(id_arrays), NUM_PERM), np.iinfo(np.uint64).max, dtype=np.uint64)
    for index, ids in enumerate(id_arrays):
        if ids.size == 0:
            continue
        # (len(ids), NUM_PERM) in Python ints to keep the modulus exact
        values = (ids[:, None].astype(object) * a[None, :].astype(object)
                  + b[None, :].astype(object)) % int(prime)
        signatures[index] = np.min(np.asarray(values, dtype=object), axis=0).astype(np.uint64)
    return signatures


def lsh_candidates(signatures):
    """Candidate pairs from banding; every pair is verified exactly afterwards."""
    pairs = set()
    for band in range(BANDS):
        lo = band * ROWS_PER_BAND
        buckets = defaultdict(list)
        for index, row in enumerate(signatures[:, lo:lo + ROWS_PER_BAND]):
            buckets[row.tobytes()].append(index)
        for members in buckets.values():
            if len(members) < 2 or len(members) > 200:
                continue
            for i, left in enumerate(members):
                for right in members[i + 1:]:
                    pairs.add((left, right) if left < right else (right, left))
    return pairs


class Union:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[max(rx, ry)] = min(rx, ry)


def parse_rating(annotation):
    """Only a valid ordinal 1..5 counts. Anything else is missing, never zero."""
    if not isinstance(annotation, dict):
        return None
    raw = annotation.get("Rating")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        value = float(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        if not re.fullmatch(r"[1-5](\.0+)?", text):
            return None
        value = float(text)
    else:
        return None
    return int(value) if 1 <= value <= 5 and float(value).is_integer() else None


def read_rows():
    for source in SOURCES:
        path = SNAPSHOT / f"{source}.jsonl"
        with path.open() as stream:
            for line_no, line in enumerate(stream):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("source") != source:
                    raise ValueError(f"{path}:{line_no} declares source {row.get('source')!r}")
                yield source, line_no, row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-name", default="v1")
    args = ap.parse_args()
    start = time.monotonic()
    out = ROOT / "splits" / args.out_name
    out.mkdir(parents=True, exist_ok=False)

    records, per_source_raw = [], Counter()
    for source, line_no, row in read_rows():
        per_source_raw[source] += 1
        instruction = row.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            continue
        norm = normalize(instruction)
        if not norm:
            continue
        completions = row.get("completions") or []
        ratings = []
        for completion in completions:
            annotations = completion.get("annotations") or {}
            ratings.append({obj: parse_rating(annotations.get(obj)) for obj in OBJECTIVES})
        records.append({"source": source, "line_no": line_no, "instruction": instruction,
                        "norm_text": norm, "norm_sha256": sha_text(norm),
                        "n_completions": len(completions), "ratings": ratings,
                        "models": row.get("models")})
    print(json.dumps({"phase": "read", "per_source_raw": dict(per_source_raw),
                      "usable_rows": len(records)}), flush=True)

    # ---- duplicate groups: exact normalized text first, then verified near-dup
    exact = defaultdict(list)
    for index, record in enumerate(records):
        exact[record["norm_sha256"]].append(index)
    reps = sorted(exact)                       # deterministic representative order
    rep_index = {key: i for i, key in enumerate(reps)}
    rep_norm = [records[exact[key][0]]["norm_text"] for key in reps]
    rep_shingles = [shingles(text) for text in rep_norm]
    signatures = minhash(shingle_ids(rep_shingles))
    candidates = lsh_candidates(signatures)
    union = Union(len(reps))
    verified = 0
    for left, right in candidates:
        a, b = rep_shingles[left], rep_shingles[right]
        if not a or not b:
            continue
        jaccard = len(a & b) / len(a | b)
        if jaccard >= NEAR_DUP_JACCARD:
            union.union(left, right)
            verified += 1
    groups = defaultdict(list)
    for key in reps:
        groups[union.find(rep_index[key])].append(key)
    print(json.dumps({"phase": "dedup", "exact_groups": len(reps),
                      "lsh_candidate_pairs": len(candidates),
                      "verified_near_dup_pairs": verified,
                      "prompt_groups": len(groups)}), flush=True)

    # ---- decontamination against the frozen evaluation bank
    bank_norm, bank_by_sha = [], {}
    with (ROOT / "data" / "eval_prompt_bank.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            bank_by_sha[row["norm_sha256"]] = row["benchmark"]
            bank_norm.append((row["benchmark"], row["norm_text"]))
    bank_shingles = [shingles(text) for _, text in bank_norm]
    bank_sig = minhash(shingle_ids(bank_shingles), seed=SPLIT_SEED + 1)
    combined = np.vstack([signatures, bank_sig])
    offset = len(reps)
    near_hits = defaultdict(set)
    for left, right in lsh_candidates(combined):
        if (left < offset) == (right < offset):
            continue
        uf_i, bank_i = (left, right - offset) if left < offset else (right, left - offset)
        a, b = rep_shingles[uf_i], bank_shingles[bank_i]
        if not a or not b:
            continue
        if len(a & b) / len(a | b) >= NEAR_DUP_JACCARD:
            near_hits[uf_i].add(bank_norm[bank_i][0])

    contaminated, contamination_rows = {}, []
    for root, members in groups.items():
        hits = set()
        for key in members:
            if key in bank_by_sha:
                hits.add((bank_by_sha[key], "exact"))
            for benchmark in near_hits.get(rep_index[key], ()):
                hits.add((benchmark, "near_dup>=0.90"))
        if hits:
            contaminated[root] = sorted(hits)
            contamination_rows.append({"group_root": int(root), "n_exact_keys": len(members),
                                       "matches": [{"benchmark": b, "rule": r} for b, r in sorted(hits)]})
    print(json.dumps({"phase": "decontaminate", "removed_groups": len(contaminated),
                      "by_benchmark": dict(Counter(b for hits in contaminated.values()
                                                   for b, _ in hits))}), flush=True)

    # ---- also drop anything already used by the SafeRLHF panel
    safe_norms = set()
    safe_dir = Path("/work/nbpo_repair_20260909/splits")
    for split in ("train", "dev", "test"):
        path = safe_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                safe_norms.add(sha_text(normalize(row["prompt"])))
    safe_overlap = {root for root, members in groups.items()
                    if any(key in safe_norms for key in members)}
    print(json.dumps({"phase": "saferlhf_overlap", "groups": len(safe_overlap),
                      "safe_prompts_read": len(safe_norms)}), flush=True)

    # ---- eligibility: every objective needs at least two rated completions
    eligible, coverage_hist = [], Counter()
    per_objective_short = Counter()
    for root, members in sorted(groups.items()):
        if root in contaminated or root in safe_overlap:
            continue
        rows = [records[i] for key in members for i in exact[key]]
        rows.sort(key=lambda r: (SOURCES.index(r["source"]), r["line_no"]))
        head = rows[0]
        valid = {obj: sum(1 for rating in head["ratings"] if rating[obj] is not None)
                 for obj in OBJECTIVES}
        worst = min(valid.values())
        coverage_hist[worst] += 1
        if worst < 2:
            for obj, count in valid.items():
                if count < 2:
                    per_objective_short[obj] += 1
            continue
        eligible.append({"prompt_id": head["norm_sha256"], "group_root": int(root),
                         "source": head["source"], "line_no": head["line_no"],
                         "instruction": head["instruction"], "norm_text": head["norm_text"],
                         "n_completions": head["n_completions"],
                         "valid_ratings_per_objective": valid,
                         "n_exact_variants": len(members), "n_rows_in_group": len(rows)})
    print(json.dumps({"phase": "eligibility", "eligible_groups": len(eligible),
                      "worst_objective_valid_count_histogram": dict(sorted(coverage_hist.items())),
                      "groups_short_by_objective": dict(per_objective_short)}), flush=True)

    # ---- split: pre-declared sizes, pre-declared fallback, stratified by source
    total_needed = sum(TARGET.values())
    if len(eligible) >= total_needed:
        sizes, regime = dict(TARGET), "target"
    elif len(eligible) >= sum(FALLBACK.values()):
        sizes, regime = dict(FALLBACK), "fallback_policy_train_5000"
    else:
        sizes, regime = None, "insufficient"
    report = {"dataset": "openbmb/UltraFeedback", "revision": DATASET_REVISION,
              "source_files": {source: {"path": str(SNAPSHOT / f"{source}.jsonl"),
                                        "sha256": file_hash(SNAPSHOT / f"{source}.jsonl"),
                                        "rows": per_source_raw[source]} for source in SOURCES},
              "objectives": list(OBJECTIVES), "split_seed": SPLIT_SEED,
              "normalization": "NFKC, curly quotes folded, whitespace collapsed, lowercased",
              "near_duplicate_rule": {"shingle_words": SHINGLE, "num_perm": NUM_PERM,
                                      "bands": BANDS, "rows_per_band": ROWS_PER_BAND,
                                      "verified_jaccard_threshold": NEAR_DUP_JACCARD},
              "raw_rows": dict(per_source_raw), "usable_rows": len(records),
              "exact_normalized_prompts": len(reps), "prompt_groups": len(groups),
              "decontaminated_groups": len(contaminated),
              "saferlhf_overlap_groups": len(safe_overlap),
              "eligible_groups": len(eligible),
              "eligibility_rule": "at least two completions with a valid ordinal 1..5 rating for EVERY objective",
              "worst_objective_valid_count_histogram": dict(sorted(coverage_hist.items())),
              "regime": regime, "sizes": sizes,
              "eval_bank_sha256": file_hash(ROOT / "data" / "eval_prompt_bank.jsonl"),
              "source_sha256": file_hash(__file__)}

    if sizes is None:
        report["status"] = "insufficient_eligible_prompts"
        report["shortfall"] = sum(FALLBACK.values()) - len(eligible)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": "INSUFFICIENT", "eligible": len(eligible),
                          "fallback_needs": sum(FALLBACK.values())}), flush=True)
        return

    rng = np.random.default_rng(SPLIT_SEED)
    by_source = defaultdict(list)
    for row in eligible:
        by_source[row["source"]].append(row)
    for source in by_source:
        by_source[source].sort(key=lambda r: r["prompt_id"])
        rng.shuffle(by_source[source])
    order = []
    # Round-robin over sources in proportion, so every split keeps the source mix.
    counts = {source: len(rows) for source, rows in by_source.items()}
    cursors = {source: 0 for source in by_source}
    total = sum(counts.values())
    quota = {source: counts[source] / total for source in by_source}
    while len(order) < sum(sizes.values()):
        remaining = {s: counts[s] - cursors[s] for s in by_source}
        if not any(remaining.values()):
            break
        source = max(remaining, key=lambda s: (quota[s] * (len(order) + 1) - cursors[s],
                                               remaining[s], s))
        if remaining[source] <= 0:
            source = max(remaining, key=lambda s: (remaining[s], s))
        order.append(by_source[source][cursors[source]])
        cursors[source] += 1

    assigned, cursor = {}, 0
    for name in ("pm_train", "pm_dev", "policy_train", "policy_dev", "final_eval"):
        rows = order[cursor:cursor + sizes[name]]
        cursor += sizes[name]
        assigned[name] = rows
        path = out / f"{name}.jsonl"
        with path.open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    ids = {name: {row["prompt_id"] for row in rows} for name, rows in assigned.items()}
    for left in ids:
        for right in ids:
            if left < right and ids[left] & ids[right]:
                raise ValueError(f"Splits {left} and {right} intersect")
    report["splits"] = {name: {"n": len(rows), "path": str(out / f"{name}.jsonl"),
                               "sha256": file_hash(out / f"{name}.jsonl"),
                               "by_source": dict(Counter(row["source"] for row in rows))}
                        for name, rows in assigned.items()}
    report["unassigned_eligible"] = len(eligible) - sum(sizes.values())
    _assigned_ids = {row["prompt_id"] for row in order}
    _unassigned = sorted((row for row in eligible
                          if row["prompt_id"] not in _assigned_ids),
                         key=lambda r: r["prompt_id"])
    with (out / "unassigned.jsonl").open("x") as stream:
        for row in _unassigned:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report["seconds"] = time.monotonic() - start
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out / "decontamination.jsonl").open("x") as stream:
        for row in contamination_rows:
            stream.write(json.dumps(row) + "\n")
    print(json.dumps({"status": "OK", "regime": regime,
                      "splits": {k: v["n"] for k, v in report["splits"].items()},
                      "unassigned_eligible": report["unassigned_eligible"],
                      "seconds": round(report["seconds"], 1)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
