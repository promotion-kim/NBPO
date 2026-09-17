"""Freeze the UW panel of the union protocol (Appendix app:union-panels).

Declared rules implemented here, not paraphrased:

  * grouping key: NFKC, normalized line endings, collapsed whitespace, over the
    complete user prompt; the original text is what the model later sees;
  * ordering: SHA256 of "nbpo-union-v1:" + panel ID + grouping key;
  * cross-play groups are reserved BEFORE train and dev are allocated;
  * four checklist items per prompt, chosen by sorting SHA256 of
    "nbpo-union-v1:item:" + item text, with no alignment of item position across
    prompts;
  * groups appearing in any supplied historical manifest, or in a downstream
    evaluation question set, are excluded before anything is measured;
  * near-duplicate audit over character 5-gram Jaccard at .90, reported;
  * planned, eligible and finally used counts are stored separately.

The requested budget is a reduced-budget protocol: the appendix asks for
2000/500/500 and this freeze takes 600/150/200, because the judging cost is
linear in prompts and the instruction is to finish as fast as possible with one
seed. The reduction is recorded in the artifact as `budget_profile` so no reader
can mistake it for the full plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

SUB = Path("/work/sub_20260914")
DATA = SUB / "data/wildchecklists/train.jsonl"
NS = "nbpo-union-v1:"
PANEL = "UW"
ITEM = re.compile(r"^\s*(\d+)\)\s*(.+?)\s*(?:\(importance:\s*(\d+)\s*/\s*100\s*\))?\s*$")
MIN_ITEMS, MAX_ITEMS = 4, 12       # four are kept, so four must exist
MIN_CHARS, MAX_CHARS = 16, 4000
ITEMS_KEPT = 4


def digest(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def file_hash(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def group_key(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", t).strip()


def shingles(t: str, n: int = 5):
    s = re.sub(r"\s+", " ", t.lower())
    return {s[i:i + n] for i in range(max(len(s) - n + 1, 1))}


def parse_items(requirements):
    items, dropped = [], 0
    for raw in (requirements or "").split("\n"):
        if not raw.strip():
            continue
        m = ITEM.match(raw)
        if not m:
            dropped += 1
            continue
        text = m.group(2).strip()
        if not text.endswith("?"):
            dropped += 1
            continue
        items.append({"text": text,
                      "importance": int(m.group(3)) if m.group(3) else None})
    return items, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=int, default=600)
    ap.add_argument("--dev", type=int, default=150)
    ap.add_argument("--crossplay", type=int, default=200)
    ap.add_argument("--planned", nargs=3, type=int, default=[2000, 500, 500],
                    help="the appendix budget, recorded for comparison")
    ap.add_argument("--tag", default="uw_v1")
    ap.add_argument("--jaccard", type=float, default=0.90)
    ap.add_argument("--slack", type=int, default=400,
                    help="extra groups drawn so near-duplicate exclusion still "
                         "meets the requested counts")
    args = ap.parse_args()

    # every historical manifest and evaluation question set this panel must avoid
    excluded_sources = []
    exclude_keys = set()
    for path in sorted(SUB.glob("panel/wild*.jsonl")) + sorted(SUB.glob("panel/eval_*.jsonl")):
        n = 0
        for line in path.open():
            if line.strip():
                row = json.loads(line)
                text = row.get("instruction") or row.get("prompt") or ""
                if text:
                    exclude_keys.add(group_key(text)); n += 1
        excluded_sources.append({"path": str(path), "rows": n,
                                 "sha256": file_hash(path)})

    counts = {"rows": 0, "rejected_length": 0, "rejected_item_count": 0,
              "rejected_duplicate_group": 0, "rejected_historical_overlap": 0,
              "dropped_item_lines": 0}
    seen, rows = set(), []
    for line in DATA.open():
        if not line.strip():
            continue
        d = json.loads(line)
        counts["rows"] += 1
        prompt = d.get("prompt") or ""
        key = group_key(prompt)
        if not (MIN_CHARS <= len(key) <= MAX_CHARS):
            counts["rejected_length"] += 1; continue
        if key in seen:
            counts["rejected_duplicate_group"] += 1; continue
        if key in exclude_keys:
            counts["rejected_historical_overlap"] += 1; continue
        items, dropped = parse_items(d.get("requirements"))
        counts["dropped_item_lines"] += dropped
        if not (MIN_ITEMS <= len(items) <= MAX_ITEMS):
            counts["rejected_item_count"] += 1; continue
        seen.add(key)
        chosen = sorted(items, key=lambda it: digest(NS + "item:" + it["text"]))[:ITEMS_KEPT]
        rows.append({"prompt_id": digest(NS + PANEL + ":" + key)[:16],
                     "instruction": prompt.strip(),
                     "group_key_sha256": digest(key),
                     "source": "wildchecklists/train",
                     "items": chosen,
                     "_key": key})

    ordered = sorted(rows, key=lambda r: (digest(NS + PANEL + ":" + r["_key"]), r["prompt_id"]))
    need = args.crossplay + args.train + args.dev
    if len(ordered) < need:
        raise SystemExit("only %d eligible groups for a request of %d" % (len(ordered), need))

    # near-duplicate exclusion BEFORE allocation: keep the earliest of each
    # colliding set in hash order, and draw from a prefix wide enough to refill
    window = ordered[:need + args.slack]
    kept, kept_grams, dropped_pairs = [], [], []
    for row in window:
        g = shingles(row["_key"])
        hit = None
        for other, og in zip(kept, kept_grams):
            inter = len(g & og)
            if not inter:
                continue
            jac = inter / len(g | og)
            if jac >= args.jaccard:
                hit = (other["prompt_id"], round(jac, 4))
                break
        if hit is None:
            kept.append(row); kept_grams.append(g)
        elif len(dropped_pairs) < 50:
            dropped_pairs.append([row["prompt_id"], hit[0], hit[1]])
        else:
            dropped_pairs.append(None)
    near = sum(1 for d in dropped_pairs if d is not None or d is None)
    pairs = [d for d in dropped_pairs if d is not None]
    if len(kept) < need:
        raise SystemExit("after near-duplicate exclusion only %d of %d groups remain; "
                         "raise --slack" % (len(kept), need))
    ordered = kept
    # cross-play is reserved first, as the appendix requires
    splits = {"crossplay": ordered[:args.crossplay],
              "train": ordered[args.crossplay:args.crossplay + args.train],
              "dev": ordered[args.crossplay + args.train:need]}
    used = [r for part in splits.values() for r in part]

    out = SUB / "panel" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, part in splits.items():
        path = out / ("%s.jsonl" % name)
        with path.open("w") as stream:
            for r in part:
                stream.write(json.dumps({k: v for k, v in r.items() if k != "_key"},
                                        ensure_ascii=False) + "\n")
        written[name] = {"prompts": len(part), "sha256": file_hash(path),
                         "path": str(path)}

    report = {
        "panel": PANEL, "tag": args.tag,
        "budget_profile": {
            "appendix_request": {"train": args.planned[0], "dev": args.planned[1],
                                 "crossplay": args.planned[2]},
            "this_freeze": {"train": args.train, "dev": args.dev,
                            "crossplay": args.crossplay},
            "why_reduced": ("judging cost is linear in prompts and the instruction is to "
                            "finish as fast as possible with one seed; this is a "
                            "reduced-budget protocol, not the appendix budget"),
        },
        "dataset": "viswavi/wildchecklists (train)", "dataset_file": str(DATA),
        "dataset_sha256": file_hash(DATA),
        "eligible_groups": len(ordered), "used_groups": len(used),
        "counts": counts, "splits": written,
        "items_per_prompt": ITEMS_KEPT,
        "item_selection": "sort SHA256('%sitem:' + item text), keep the first four" % NS,
        "no_position_alignment": ("item k of two prompts is never one objective; the four "
                                  "items are that prompt's own"),
        "group_rule": ("NFKC, normalized line endings, collapsed whitespace over the whole "
                       "user prompt"),
        "order_rule": "sort by SHA256('%s' + panel + ':' + group key)" % NS,
        "crossplay_reserved_first": True,
        "historical_exclusion": {"sources": excluded_sources,
                                 "excluded_groups": counts["rejected_historical_overlap"]},
        "near_duplicate_exclusion": {
            "threshold": args.jaccard,
            "groups_dropped": near,
            "rule": "keep the earliest of each colliding set in hash order",
            "before_allocation": True,
            "examples": pairs},
        "seeds": {"policy": [42], "note": "one seed by instruction; 43/44 not run"},
        "source_sha256": file_hash(__file__),
    }
    (out / "freeze.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("tag", "eligible_groups", "used_groups", "counts", "splits",
                       "near_duplicate_exclusion")}, ensure_ascii=False)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
