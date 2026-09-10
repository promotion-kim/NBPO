"""Reserve and freeze the within-rubric audit prompts (Appendix C).

100 audit prompts plus a separate 20-prompt pilot, drawn from eligible UF-4
prompt groups that were left unassigned by the five frozen splits, stratified by
source. The pilot exists to measure parsing and throughput and is excluded from
every audit estimate.

Disjointness is verified against all five split files and the frozen benchmark
bank rather than assumed, and the IDs are written once: this file refuses to
overwrite an existing reservation, because the freeze has to precede any
judgment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/work/uf4_20260910")
SNAPSHOT = Path("/work/hf_cache/hub/datasets--openbmb--UltraFeedback/snapshots/"
                "40b436560ca83a8dba36114c22ab3c66e43f6d5e")
SOURCES = ("ultrachat", "sharegpt", "evol_instruct")
OBJECTIVES = ("instruction_following", "truthfulness", "honesty", "helpfulness")
RESERVE_SEED = 20260911


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def parse_rating(annotation):
    if not isinstance(annotation, dict):
        return None
    raw = annotation.get("Rating")
    if isinstance(raw, str) and re.fullmatch(r"[1-5](\.0+)?", raw.strip()):
        return int(float(raw.strip()))
    if isinstance(raw, (int, float)) and not isinstance(raw, bool) and 1 <= float(raw) <= 5:
        return int(raw) if float(raw).is_integer() else None
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", default="v1")
    ap.add_argument("--out-name", default="v1")
    ap.add_argument("--n-audit", type=int, default=100)
    ap.add_argument("--n-pilot", type=int, default=20)
    args = ap.parse_args()

    out = ROOT / "audit" / args.out_name
    out.mkdir(parents=True, exist_ok=False)
    split_dir = ROOT / "splits" / args.splits

    used, split_hashes = set(), {}
    for split in ("pm_train", "pm_dev", "policy_train", "policy_dev", "final_eval"):
        path = split_dir / f"{split}.jsonl"
        split_hashes[split] = file_hash(path)
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    used.add(json.loads(line)["prompt_id"])
    bank = set()
    with (ROOT / "data/eval_prompt_bank.jsonl").open() as stream:
        for line in stream:
            bank.add(json.loads(line)["norm_sha256"])

    # Rebuild the eligible pool the same way the split builder did, then keep
    # only groups no split claimed.
    candidates = []
    for source in SOURCES:
        with (SNAPSHOT / f"{source}.jsonl").open() as stream:
            for line_no, line in enumerate(stream):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                instruction = row.get("instruction")
                if not isinstance(instruction, str) or not instruction.strip():
                    continue
                norm = normalize(instruction)
                pid = sha_text(norm)
                if not norm or pid in used or pid in bank:
                    continue
                completions = row.get("completions") or []
                if len(completions) < 4:
                    continue
                valid = {o: 0 for o in OBJECTIVES}
                for completion in completions:
                    ann = completion.get("annotations") or {}
                    for o in OBJECTIVES:
                        if parse_rating(ann.get(o)) is not None:
                            valid[o] += 1
                if min(valid.values()) < 2:
                    continue
                candidates.append({"prompt_id": pid, "source": source, "line_no": line_no,
                                   "instruction": instruction, "norm_text": norm})
    seen, unique = set(), []
    for c in candidates:                       # one entry per prompt group
        if c["prompt_id"] in seen:
            continue
        seen.add(c["prompt_id"])
        unique.append(c)

    by_source = defaultdict(list)
    for c in unique:
        by_source[c["source"]].append(c)
    rng = np.random.default_rng(RESERVE_SEED)
    for source in by_source:
        by_source[source].sort(key=lambda r: r["prompt_id"])
        rng.shuffle(by_source[source])

    total = args.n_audit + args.n_pilot
    share = {s: len(by_source[s]) / len(unique) for s in by_source}
    picked, cursors = [], {s: 0 for s in by_source}
    while len(picked) < total:
        source = max(by_source, key=lambda s: (share[s] * (len(picked) + 1) - cursors[s],
                                               len(by_source[s]) - cursors[s], s))
        if cursors[source] >= len(by_source[source]):
            source = max(by_source, key=lambda s: len(by_source[s]) - cursors[s])
        picked.append(by_source[source][cursors[source]])
        cursors[source] += 1
    pilot, audit = picked[:args.n_pilot], picked[args.n_pilot:]
    if len({p["prompt_id"] for p in picked}) != total:
        raise ValueError("Reserved prompts are not distinct")
    if {p["prompt_id"] for p in picked} & used:
        raise ValueError("A reserved prompt is already in a frozen split")

    for name, rows in (("audit_100", audit), ("pilot_20", pilot)):
        path = out / f"{name}.jsonl"
        with path.open("x") as stream:
            for r in rows:
                stream.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "protocol": "Appendix C within-objective cyclic-preference audit (app:within-rubric-audit)",
        "reserve_seed": RESERVE_SEED,
        "eligible_unassigned_groups": len(unique),
        "n_audit": len(audit), "n_pilot": len(pilot),
        "audit_by_source": dict(Counter(r["source"] for r in audit)),
        "pilot_by_source": dict(Counter(r["source"] for r in pilot)),
        "disjointness_checked_against": {"splits": split_hashes,
                                         "benchmark_bank_sha256": file_hash(ROOT / "data/eval_prompt_bank.jsonl")},
        "pilot_excluded_from_estimates": True,
        "files": {name: {"path": str(out / f"{name}.jsonl"),
                         "sha256": file_hash(out / f"{name}.jsonl")}
                  for name in ("audit_100", "pilot_20")},
        "frozen_before_any_judgment": True,
        "source_sha256": file_hash(__file__)}
    (out / "reservation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("disjointness_checked_against", "files")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
