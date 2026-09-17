"""Freeze the WildChecklists panel for Table 37's Wild row.

The contract uses this source for a NATIVE item cycle audit only: its checklist
items are prompt-specific, so they are never pooled as global objectives and the
row's gamma* and TV columns stay inapplicable. Here we fix which prompts and
which items the audit covers.

Item parsing. `requirements` is a numbered text block, and not every numbered
line is a judgeable criterion: some are stems that end in a colon and renumber
their sub-points from 1, so a naive parse both double-counts and produces lines
a judge cannot answer. Only question-form lines are kept -- the text must end
in a question mark -- and everything dropped is counted in the report rather
than discarded silently. Importance weights are recorded but not used: this
audit measures cycles within an item, not a weighted aggregate.

Prompt selection is a namespaced hash of the normalized prompt, so no score,
checklist length or preference label enters it. Rows whose prompt collides
after normalization are deduplicated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

SUB = Path("/work/sub_20260914")
DATA = SUB / "data/wildchecklists/train.jsonl"
NS_PANEL = "sub_20260914-wild-panel:"
ITEM = re.compile(r"^\s*(\d+)\)\s*(.+?)\s*(?:\(importance:\s*(\d+)\s*/\s*100\s*\))?\s*$")
MIN_ITEMS, MAX_ITEMS = 2, 12
MIN_CHARS, MAX_CHARS = 16, 4000


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


def parse_items(requirements):
    """(judgeable items, dropped line count). Question form only."""
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
            dropped += 1          # a stem such as "... the following two criteria:"
            continue
        items.append({"text": text,
                      "importance": int(m.group(3)) if m.group(3) else None})
    return items, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--tag", default="wild200")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many prompts in the frozen hash order before "
                         "taking --n, so a training panel can be made disjoint "
                         "from the already-frozen screening panel")
    args = ap.parse_args()

    stats = Counter()
    seen, rows = set(), []
    with DATA.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            d = json.loads(line)
            stats["rows"] += 1
            prompt = d.get("prompt") or ""
            norm = normalize(prompt)
            if not (MIN_CHARS <= len(norm) <= MAX_CHARS):
                stats["rejected_length"] += 1
                continue
            if norm in seen:
                stats["rejected_duplicate_prompt"] += 1
                continue
            items, dropped = parse_items(d.get("requirements"))
            stats["dropped_lines"] += dropped
            if not (MIN_ITEMS <= len(items) <= MAX_ITEMS):
                stats["rejected_item_count"] += 1
                continue
            seen.add(norm)
            rows.append({"prompt_id": digest(NS_PANEL + norm)[:16],
                         "instruction": prompt.strip(),
                         "normalized_sha256": digest(norm),
                         "source": "wildchecklists/train",
                         "items": items})
            stats["eligible"] += 1

    ordered = sorted(rows, key=lambda r: (digest(NS_PANEL + r["normalized_sha256"]),
                                          r["prompt_id"]))
    panel = ordered[args.offset:args.offset + args.n]
    if len(panel) < args.n:
        raise SystemExit("only %d eligible prompts" % len(panel))

    panel_path = SUB / "panel" / ("%s.jsonl" % args.tag)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    if panel_path.exists():
        print(json.dumps({"skipped": "panel already frozen", "path": str(panel_path)}))
        return 0
    with panel_path.open("x") as stream:
        for r in panel:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")

    per_prompt = [len(r["items"]) for r in panel]
    total_items = sum(per_prompt)
    report = {
        "tag": args.tag,
        "dataset": "viswavi/wildchecklists",
        "dataset_file": str(DATA), "dataset_sha256": file_hash(DATA),
        "dataset_revision_note": ("downloaded from the hub at main on 2026-09-15; the "
                                  "repository exposes one train.jsonl and no dated "
                                  "revision, so this file hash is the identifier"),
        "prompts": len(panel), "items_total": total_items,
        "items_per_prompt": {"mean": round(total_items / len(panel), 2),
                             "min": min(per_prompt), "max": max(per_prompt)},
        "eligibility": {"item_count_range": [MIN_ITEMS, MAX_ITEMS],
                        "normalized_char_range": [MIN_CHARS, MAX_CHARS],
                        "items_kept": "numbered lines in question form",
                        "items_dropped": "numbered lines that are stems, not questions"},
        "counts": dict(stats),
        "selection_rule": "sort eligible unique prompts by SHA256('%s' + normalized prompt)" % NS_PANEL,
        "offset": args.offset,
        "disjointness": ("prompts [offset, offset+n) of the frozen order; offset 200 is "
                         "disjoint from the wild200 screening panel by construction"),
        "label_free": "no score, preference label or checklist length enters the sort key",
        "importance_weights": "recorded per item, not used: this audit is within-item",
        "verdict_budget": {"pairs": 28, "orders": 2, "repeats": 2,
                           "items": total_items,
                           "total": 28 * 2 * 2 * total_items},
        "panel_sha256": file_hash(panel_path),
        "source_sha256": file_hash(__file__)}
    (SUB / "panel" / ("%s_freeze.json" % args.tag)).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("prompts", "items_total", "items_per_prompt", "counts",
                       "verdict_budget")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
