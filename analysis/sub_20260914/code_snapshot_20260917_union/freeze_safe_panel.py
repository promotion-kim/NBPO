"""Freeze the PKU-SafeRLHF audit panel before any candidate is generated or judged.

Prompts are chosen by a hash of the normalized prompt text under a declared
namespace, so the selection cannot see a label, a score, a cycle or a win rate.
Three nested panels are written in one pass -- the 200-prompt screening panel,
the 10-prompt throughput pilot inside it, and the 50-prompt confirmation set
inside it -- because choosing the confirmation set after reading screening
results would turn independent repeats into a selected re-judge of whichever
edges looked interesting.

The file refuses to overwrite an existing freeze. Everything the panel depends
on is recorded: dataset snapshot, per-source row counts, eligibility rule,
namespaces, and the sha256 of the written panel files.
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
NS_PANEL = "sub_20260914-safe-panel:"
NS_PILOT = "sub_20260914-safe-pilot:"
NS_CONFIRM = "sub_20260914-safe-confirm:"
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
    """Collapse whitespace and unicode form so near-duplicate prompts collide.

    Leakage control later in the campaign keys on this same string, so the
    audit and the training splits cannot disagree about what counts as the
    same prompt.
    """
    t = unicodedata.normalize("NFKC", text).replace("​", "")
    return re.sub(r"\s+", " ", t).strip().lower()


def eligible_rows():
    """Every train row, deduplicated by normalized prompt, with provenance kept."""
    seen, rows, stats = {}, [], Counter()
    for path in sorted(glob.glob(SNAPSHOT + "/data/*/train.jsonl")):
        source_model = Path(path).parent.name
        with open(path) as stream:
            for line in stream:
                if not line.strip():
                    continue
                d = json.loads(line)
                stats["rows_total"] += 1
                prompt = d.get("prompt") or ""
                norm = normalize(prompt)
                if not (MIN_CHARS <= len(norm) <= MAX_CHARS):
                    stats["rejected_length"] += 1
                    continue
                if norm in seen:
                    stats["rejected_duplicate_prompt"] += 1
                    seen[norm]["occurrences"] += 1
                    continue
                row = {"prompt_id": digest(NS_PANEL + norm)[:16],
                       "instruction": prompt.strip(),
                       "normalized_sha256": digest(norm),
                       "source": "PKU-SafeRLHF/" + source_model,
                       "prompt_source": d.get("prompt_source"),
                       "occurrences": 1,
                       # The original human labels are carried for later
                       # comparison but are NOT used to select anything, and
                       # they are never mixed with the new AI judgments.
                       "human_better_response_id": d.get("better_response_id"),
                       "human_safer_response_id": d.get("safer_response_id"),
                       "human_is_response_0_safe": d.get("is_response_0_safe"),
                       "human_is_response_1_safe": d.get("is_response_1_safe")}
                seen[norm] = row
                rows.append(row)
                stats["eligible_unique"] += 1
    return rows, stats


def pick(rows, namespace, n):
    """The n smallest hashes under one namespace, ties broken by prompt id."""
    keyed = sorted(rows, key=lambda r: (digest(namespace + r["normalized_sha256"]),
                                        r["prompt_id"]))
    return keyed[:n]


def write_jsonl(path: Path, rows) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        for r in rows:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    return file_hash(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-screen", type=int, default=200)
    ap.add_argument("--n-pilot", type=int, default=10)
    ap.add_argument("--n-confirm", type=int, default=50)
    args = ap.parse_args()

    panel_dir = ROOT / "panel"
    if (panel_dir / "screen200.jsonl").exists():
        print(json.dumps({"skipped": "panel already frozen",
                          "path": str(panel_dir / "screen200.jsonl")}))
        return 0

    rows, stats = eligible_rows()
    screen = pick(rows, NS_PANEL, args.n_screen)
    pilot = pick(screen, NS_PILOT, args.n_pilot)
    confirm = pick(screen, NS_CONFIRM, args.n_confirm)

    pilot_ids = {r["prompt_id"] for r in pilot}
    confirm_ids = {r["prompt_id"] for r in confirm}
    screen_ids = {r["prompt_id"] for r in screen}
    assert pilot_ids <= screen_ids and confirm_ids <= screen_ids
    assert len(screen_ids) == args.n_screen

    hashes = {"screen200.jsonl": write_jsonl(panel_dir / "screen200.jsonl", screen),
              "pilot10.jsonl": write_jsonl(panel_dir / "pilot10.jsonl", pilot),
              "confirm50.jsonl": write_jsonl(panel_dir / "confirm50.jsonl", confirm)}

    manifest = {
        "experiment_id": "sub_20260914",
        "dataset": "PKU-Alignment/PKU-SafeRLHF",
        "dataset_snapshot": SNAPSHOT.rsplit("/", 1)[-1],
        "dataset_license": "cc-by-nc-4.0 (README of the cached snapshot)",
        "split_used": "train (all three response-source configs)",
        "objectives": ["helpfulness", "harmlessness"],
        "eligibility": {"single_field_prompt": True,
                        "normalized_char_range": [MIN_CHARS, MAX_CHARS],
                        "deduplicated_by": "NFKC + whitespace-collapsed lowercase prompt"},
        "selection_rule": {
            "screen": "sort eligible unique prompts by SHA256('%s' + normalized_prompt)" % NS_PANEL,
            "pilot": "sort the screening panel by SHA256('%s' + normalized_prompt)" % NS_PILOT,
            "confirmation": "sort the screening panel by SHA256('%s' + normalized_prompt)" % NS_CONFIRM,
            "label_free": ("no field of the human annotation, no model score and no judgment "
                           "enters the sort key")},
        "counts": {"screen": len(screen), "pilot": len(pilot), "confirmation": len(confirm),
                   **dict(stats)},
        "sources": dict(Counter(r["source"] for r in screen)),
        "panel_sha256": hashes,
        "human_labels": ("carried in each row for later comparison with the new AI judgments; "
                         "never merged with them and never used for selection"),
        "source_sha256": file_hash(__file__)}
    (panel_dir / "freeze.json").write_text(json.dumps(manifest, indent=2,
                                                      ensure_ascii=False) + "\n")
    print(json.dumps(manifest, ensure_ascii=False)[:1400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
