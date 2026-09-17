"""Freeze the UT panel of the union protocol (Appendix app:union-panels).

Declared rules implemented here, not paraphrased:

  * data is openai/summarize_from_feedback, comparisons; train/dev come from the
    official train split and test only from the official validation split
    (valid1 and valid2 together, which is what that dataset's own loader calls
    VALIDATION), as the panel table requires;
  * only Reddit examples with a nonempty info.post are eligible; the CNN/DM
    rows, which carry info.article instead, are ineligible and are NOT replaced
    by a different summarization dataset;
  * a summary's grouping key is its source-post ID plus the normalized source
    text, exactly as the grouping paragraph states;
  * the instruction is the declared string "Summarize the following Reddit post
    in one sentence.", followed by info.title when present and then the post;
  * two objectives, whose complete rubric strings and hashes are frozen here
    before any judgment is made: overall summary quality/coverage, and whether
    every asserted fact is supported by the source;
  * groups appearing in any supplied historical manifest, or in a downstream
    evaluation question set, are excluded before anything is measured, and the
    two splits are also made disjoint under the grouping key;
  * near-duplicate audit over character 5-gram Jaccard at .90, applied before
    allocation and reported;
  * planned, eligible and finally used counts are stored separately.

The requested budget is a reduced-budget protocol: the appendix asks for
2000/500 train/dev and 1000 test, and this freeze takes 600/150/250, because
judging cost is linear in prompts and the instruction is to finish as fast as
possible with one seed. The reduction is recorded as `budget_profile`.

The released human comparisons are NOT used. This panel supplies source posts
only; the two objectives are the declared AI rubrics below.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

SUB = Path("/work/sub_20260914")
DATA = SUB / "data/tldr_comparisons/unique_posts.jsonl"
NS = "nbpo-union-v1:"
PANEL = "UT"
INSTRUCTION = "Summarize the following Reddit post in one sentence."
MIN_CHARS, MAX_CHARS = 200, 6000

OBJECTIVES = [
    {"objective_id": "pref",
     "name": "Summary preference",
     "text": ("Does the summary read as the better one-sentence summary of this "
              "post overall, covering what matters most in it?")},
    {"objective_id": "faith",
     "name": "Source faithfulness",
     "text": ("Is every fact asserted by the summary supported by the source "
              "post, with nothing added that the post does not say?")},
]


def digest(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def file_hash(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", t).strip()


def group_key(post_id: str, post: str) -> str:
    """The declared key: source-post ID plus normalized source text."""
    return "%s\n%s" % (post_id, norm(post))


def shingles(t: str, n: int = 5):
    s = re.sub(r"\s+", " ", t.lower())
    return {s[i:i + n] for i in range(max(len(s) - n + 1, 1))}


def render_prompt(title: str, post: str) -> str:
    if title.strip():
        return "%s\n\nTitle: %s\n\n%s" % (INSTRUCTION, title.strip(), post.strip())
    return "%s\n\n%s" % (INSTRUCTION, post.strip())


def dedup(window, threshold):
    kept, grams, pairs, dropped = [], [], [], 0
    for row in window:
        g = shingles(norm(row["_post"]))
        hit = None
        for other, og in zip(kept, grams):
            inter = len(g & og)
            if not inter:
                continue
            j = inter / len(g | og)
            if j >= threshold:
                hit = (other["prompt_id"], round(j, 4))
                break
        if hit is None:
            kept.append(row); grams.append(g)
        else:
            dropped += 1
            if len(pairs) < 50:
                pairs.append([row["prompt_id"], hit[0], hit[1]])
    return kept, dropped, pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=int, default=600)
    ap.add_argument("--dev", type=int, default=150)
    ap.add_argument("--test", type=int, default=250)
    ap.add_argument("--planned", nargs=3, type=int, default=[2000, 500, 1000])
    ap.add_argument("--tag", default="ut_v1")
    ap.add_argument("--jaccard", type=float, default=0.90)
    ap.add_argument("--slack", type=int, default=600)
    args = ap.parse_args()

    excluded_sources, exclude_keys = [], set()
    for path in (sorted(SUB.glob("panel/tldr*.jsonl")) + sorted(SUB.glob("panel/eval_*.jsonl"))
                 + sorted(SUB.glob("panel/*/crossplay.jsonl"))):
        n = 0
        for line in path.open():
            if line.strip():
                row = json.loads(line)
                text = row.get("instruction") or row.get("prompt") or ""
                if text:
                    exclude_keys.add(norm(text)); n += 1
        excluded_sources.append({"path": str(path), "rows": n, "sha256": file_hash(path)})

    counts = {"rows": 0, "rejected_length": 0, "rejected_duplicate_group": 0,
              "rejected_historical_overlap": 0, "rejected_empty_post": 0}
    pools = {"train": [], "validation": []}
    seen = set()
    for line in DATA.open():
        if not line.strip():
            continue
        d = json.loads(line)
        counts["rows"] += 1
        post = d.get("post") or ""
        if not post.strip():
            counts["rejected_empty_post"] += 1
            continue
        key = group_key(d["post_id"], post)
        prompt = render_prompt(d.get("title") or "", post)
        if not (MIN_CHARS <= len(norm(prompt)) <= MAX_CHARS):
            counts["rejected_length"] += 1
            continue
        if key in seen:
            counts["rejected_duplicate_group"] += 1
            continue
        if norm(prompt) in exclude_keys:
            counts["rejected_historical_overlap"] += 1
            continue
        seen.add(key)
        bucket = "train" if d["split"] == "train" else "validation"
        pools[bucket].append({
            "prompt_id": digest(NS + PANEL + ":" + key)[:16],
            "instruction": prompt,
            "group_key_sha256": digest(key),
            "source": "summarize_from_feedback/comparisons/%s" % d["split"],
            "post_id": d["post_id"],
            "subreddit": d.get("subreddit") or "",
            "objectives": [dict(o) for o in OBJECTIVES],
            "_post": post, "_key": key})
    for bucket in pools:
        pools[bucket].sort(key=lambda r: (digest(NS + PANEL + ":" + r["_key"]),
                                          r["prompt_id"]))

    tr_keys = {r["_key"] for r in pools["train"]}
    before = len(pools["validation"])
    pools["validation"] = [r for r in pools["validation"] if r["_key"] not in tr_keys]
    cross = before - len(pools["validation"])

    need_tr = args.train + args.dev
    if len(pools["train"]) < need_tr or len(pools["validation"]) < args.test:
        raise SystemExit("eligible train %d (need %d), validation %d (need %d)"
                         % (len(pools["train"]), need_tr,
                            len(pools["validation"]), args.test))

    tr_kept, tr_drop, tr_pairs = dedup(pools["train"][:need_tr + args.slack], args.jaccard)
    te_kept, te_drop, te_pairs = dedup(pools["validation"][:args.test + args.slack],
                                       args.jaccard)
    if len(tr_kept) < need_tr or len(te_kept) < args.test:
        raise SystemExit("after near-duplicate exclusion train %d, test %d remain; "
                         "raise --slack" % (len(tr_kept), len(te_kept)))

    splits = {"train": tr_kept[:args.train], "dev": tr_kept[args.train:need_tr],
              "test": te_kept[:args.test]}
    ids = [r["prompt_id"] for part in splits.values() for r in part]
    if len(set(ids)) != len(ids):
        raise SystemExit("prompt_id collision across the frozen splits")

    out = SUB / "panel" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, part in splits.items():
        path = out / ("%s.jsonl" % name)
        with path.open("w") as s:
            for r in part:
                s.write(json.dumps({k: v for k, v in r.items()
                                    if not k.startswith("_")},
                                   ensure_ascii=False) + "\n")
        written[name] = {"prompts": len(part), "sha256": file_hash(path),
                         "path": str(path)}

    download = json.loads((DATA.parent / "download_record.json").read_text())
    report = {
        "panel": PANEL, "tag": args.tag,
        "budget_profile": {
            "appendix_request": {"train": args.planned[0], "dev": args.planned[1],
                                 "test": args.planned[2]},
            "this_freeze": {"train": args.train, "dev": args.dev, "test": args.test},
            "why_reduced": ("judging cost is linear in prompts and the instruction is to "
                            "finish as fast as possible with one seed; this is a "
                            "reduced-budget protocol, not the appendix budget"),
        },
        "dataset": "openai/summarize_from_feedback (comparisons)",
        "dataset_access": ("the HF entry point is a loading script the installed datasets "
                           "version will not execute, so the batches were read from the "
                           "same public blob URL that script uses; per-file sha256 in "
                           "download_record.json"),
        "dataset_file": str(DATA), "dataset_sha256": file_hash(DATA),
        "download_record": str(DATA.parent / "download_record.json"),
        "source_rows_seen": download["counts"],
        "released_labels_used": False,
        "released_labels_note": ("the release's human choice between candidate summaries is "
                                 "not read; this panel supplies source posts only"),
        "cnndm_rows_ineligible": download["counts"]["cnndm_article"],
        "instruction_string": INSTRUCTION,
        "instruction_sha256": digest(INSTRUCTION),
        "prompt_rule": ("the instruction, then 'Title: <info.title>' when the title is "
                        "nonempty, then the post"),
        "objectives": [dict(o, text_sha256=digest(o["text"])) for o in OBJECTIVES],
        "eligible_groups": {"train": len(pools["train"]),
                            "validation": len(pools["validation"])},
        "used_groups": len(ids), "counts": counts,
        "cross_split_overlap_removed": cross,
        "splits": written,
        "group_rule": "source-post ID plus NFKC/collapsed-whitespace source text",
        "order_rule": "sort by SHA256('%s' + panel + ':' + group key)" % NS,
        "test_from_official_validation_split_only": True,
        "validation_definition": ("valid1 and valid2 together, which is the VALIDATION "
                                  "split of this dataset's own loader"),
        "crossplay_reserved": False,
        "crossplay_note": "the appendix reserves cross-play groups for UW only",
        "historical_exclusion": {"sources": excluded_sources,
                                 "excluded_groups": counts["rejected_historical_overlap"]},
        "near_duplicate_exclusion": {
            "threshold": args.jaccard,
            "groups_dropped": {"train": tr_drop, "validation": te_drop},
            "rule": "keep the earliest of each colliding set in hash order",
            "before_allocation": True,
            "examples": {"train": tr_pairs, "validation": te_pairs}},
        "seeds": {"policy": [42], "note": "one seed by instruction; 43/44 not run"},
        "source_sha256": file_hash(__file__),
    }
    (out / "freeze.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("tag", "eligible_groups", "used_groups", "counts", "splits",
                       "cross_split_overlap_removed", "near_duplicate_exclusion")},
                     ensure_ascii=False)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
