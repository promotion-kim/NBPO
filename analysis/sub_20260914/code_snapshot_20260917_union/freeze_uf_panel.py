"""Freeze the UF screening panel for Table 37's UF row, reusing the existing pool.

The contract says this row is only fillable under the SAME N=8, two-order,
two-repeat contract as the rest of the campaign, and that the old N=4 audit
values must not be copied into it. Nothing here copies them.

No generation is needed: pool v1 already holds eight learner occurrences per
prompt from the same base at the same decoding the Safe panel used -- T=1,
top_p=1, 1024 tokens, eight per role -- so the eight occurrences are drawn from
the same distribution as the Safe screening's eight candidates. This script
selects 200 prompts by a namespaced hash of the normalized instruction, which
cannot see a rating or a score, and materializes their learner occurrences in
the layout judge_pairs.py reads.

It also writes the four-objective rubric file: the frozen UF rubrics with the
joint_control entry removed, since Table 37's D is defined over the declared
objectives and a joint control is not one of them. The four texts are copied
byte-for-byte and the source hash is recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

UF = Path("/work/uf4_20260910")
SUB = Path("/work/sub_20260914")
NS_PANEL = "sub_20260914-uf-panel:"
OBJECTIVES = ("instruction_following", "truthfulness", "honesty", "helpfulness")


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


def load_pool(pool_name, shards=4):
    """prompt_id -> {index: learner event}, hash-verified like the solver does."""
    pool = {}
    for shard in range(shards):
        directory = UF / "pools" / pool_name / ("shard%d" % shard)
        for path in sorted(directory.glob("chunk*.jsonl")):
            manifest = json.loads((directory / (path.stem + ".manifest.json")).read_text())
            if file_hash(path) != manifest["sha256"]:
                raise ValueError("pool chunk hash mismatch: %s" % path)
            with path.open() as stream:
                for line in stream:
                    e = json.loads(line)
                    if e["role"] != "learner":
                        continue
                    pool.setdefault(e["prompt_id"], {})[e["sample_index"]] = e
    return pool


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="v1")
    ap.add_argument("--pool-shards", type=int, default=4)
    ap.add_argument("--split", default="policy_train")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--tag", default="uf_screen200")
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            (UF / "splits/v1" / ("%s.jsonl" % args.split)).open() if l.strip()]
    pool = load_pool(args.pool, args.pool_shards)
    eligible = [r for r in rows
                if r["prompt_id"] in pool and len(pool[r["prompt_id"]]) == 8]
    ordered = sorted(eligible,
                     key=lambda r: (digest(NS_PANEL + normalize(r["instruction"])),
                                    r["prompt_id"]))
    panel = ordered[:args.n]
    if len(panel) < args.n:
        raise SystemExit("only %d eligible prompts with eight occurrences" % len(panel))

    out = SUB / "responses" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "responses.jsonl"
    if dest.exists():
        print(json.dumps({"skipped": "already materialized", "path": str(dest)}))
        return 0

    events, capped = [], 0
    with dest.open("x") as stream:
        for row in panel:
            for index in range(8):
                e = pool[row["prompt_id"]][index]
                capped += int(bool(e.get("capped_horizon")))
                stream.write(json.dumps({
                    "prompt_id": row["prompt_id"], "source": row["source"],
                    "instruction": e["prompt"],
                    "response_id": "%s:%d" % (row["prompt_id"], index),
                    "response_index": index, "seed": e["seed"],
                    "response": e["response"], "n_tokens": e["n_tokens"],
                    "finish_reason": e["finish_reason"],
                    "capped": bool(e.get("capped_horizon")),
                    "response_sha256": e["response_sha256"],
                }, ensure_ascii=False) + "\n")
                events.append(e)

    # the four declared objectives, copied from the frozen UF rubric file
    source_rubrics = UF / "audit/v1/rubrics.json"
    frozen = json.loads(source_rubrics.read_text())
    rubrics = {k: frozen["rubrics"][k] for k in OBJECTIVES}
    rubric_path = SUB / "contract/rubrics_uf4.json"
    rubric_path.write_text(json.dumps({
        "rubric_set_version": "sub_20260914-uf4-four-objectives",
        "note": ("the four declared UltraFeedback objectives, copied verbatim from the "
                 "frozen UF-4 audit rubric file. The joint_control entry of that file is "
                 "excluded: Table 37's D is defined over the declared objectives and a "
                 "joint control is not one of them."),
        "source": str(source_rubrics), "source_sha256": file_hash(source_rubrics),
        "rubrics": rubrics}, indent=2, ensure_ascii=False) + "\n")

    panel_path = SUB / "panel" / ("%s.jsonl" % args.tag)
    with panel_path.open("w") as stream:
        for row in panel:
            stream.write(json.dumps({"prompt_id": row["prompt_id"],
                                     "instruction": row["instruction"],
                                     "source": row["source"]},
                                    ensure_ascii=False) + "\n")

    report = {
        "tag": args.tag, "pool": args.pool, "split": args.split,
        "prompts": len(panel), "responses": len(events),
        "eligible_prompts_in_split": len(eligible),
        "selection_rule": "sort eligible prompts by SHA256('%s' + normalized instruction)" % NS_PANEL,
        "reused_generation": ("pool %s learner occurrences: same base, T=1, top_p=1, 1024 "
                              "tokens, eight per prompt, identical to the Safe screening's "
                              "candidate decoding" % args.pool),
        "no_values_copied_from_the_old_n4_audit": True,
        "capped_responses": capped,
        "sources": dict(Counter(r["source"] for r in panel)),
        "responses_sha256": file_hash(dest),
        "panel_sha256": file_hash(panel_path),
        "rubrics_sha256": file_hash(rubric_path),
        "objectives": list(OBJECTIVES),
        "verdict_budget": {"pairs": 28, "orders": 2, "repeats": 2,
                           "objectives": len(OBJECTIVES),
                           "per_prompt": 28 * 2 * 2 * len(OBJECTIVES),
                           "total": 28 * 2 * 2 * len(OBJECTIVES) * len(panel)},
        "source_sha256": file_hash(__file__)}
    (out / "freeze.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("tag", "prompts", "responses", "eligible_prompts_in_split",
                       "capped_responses", "verdict_budget")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
