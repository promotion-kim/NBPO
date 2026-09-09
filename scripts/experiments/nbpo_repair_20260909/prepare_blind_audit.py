"""Freeze 100 outcome-independent prompts and per-prompt blinded model order.

This produces an agent-assisted qualitative audit, not human labels or a
validated refusal classifier. Source outputs are copied verbatim. Terminal
causes may only be adjudicated when raw termination evidence exists.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from scripts.experiments.nbpo_repair_20260909.common import file_hash, read_jsonl, write_json, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    labels = ["base", "canonN700", "RB1200"]
    files = {label: args.responses / label / "alpaca_eval.jsonl" for label in labels}
    rows = {label: {r["uid"]: r for r in read_jsonl(path)} for label, path in files.items()}
    if any(len(block) != 805 for block in rows.values()):
        raise ValueError("Audit source must contain all805Alpaca prompts per model")
    uids = set(rows[labels[0]])
    if any(set(block) != uids for block in rows.values()):
        raise ValueError("Prompt coverage mismatch")
    for uid in uids:
        if len({rows[label][uid]["prompt"] for label in labels}) != 1:
            raise ValueError("Prompt text mismatch")
    excluded = {"alpaca:409": "Specific non-user-provided copyrighted song verse; refusal may be appropriate",
                "alpaca:305": "Location-based non-user-provided copyrighted movie dialogue; refusal may be appropriate"}
    selected = sorted(uids - excluded.keys(), key=lambda uid: hashlib.sha256(
        ("20260909-blind100:" + rows[labels[0]][uid]["prompt"]).encode()).hexdigest())[:100]
    rng = random.Random(20260909)
    blinded, key = [], []
    for index, uid in enumerate(selected):
        order = labels.copy()
        rng.shuffle(order)
        entry = {"audit_id": index + 1, "prompt": rows[labels[0]][uid]["prompt"], "responses": {}}
        for letter, label in zip("ABC", order):
            r = rows[label][uid]
            entry["responses"][letter] = {"output": r["output"],
                "n_output_tokens": r.get("n_output_tokens"),
                "finish_reason": r.get("finish_reason", "not_saved_in_legacy_source"),
                "stop_reason": r.get("stop_reason", "not_saved_in_legacy_source")}
            key.append({"audit_id": index + 1, "uid": uid, "blind_label": letter, "model": label})
        blinded.append(entry)
    write_jsonl(args.out / "blind100.jsonl", blinded)
    write_jsonl(args.out / "unblinding_key.jsonl", key)
    write_json(args.out / "manifest.json", {"source_sha256": {label: file_hash(path) for label, path in files.items()},
        "selection": "First100 SHA256(20260909-blind100:+originalprompt), after prompt-only copyright exclusions, before response adjudication",
        "prompt_only_exclusions": excluded,
        "n_prompts": 100, "n_responses": 300,
        "benign_status": "Prompt-only screened first100 plus two replacements; reviewer to flag any ambiguous cases, without outcome-based replacement",
        "method": "Blinded coding-agent-assisted qualitative review; not human/official labels",
        "blind_file_sha256": file_hash(args.out / "blind100.jsonl"), "source_code_sha256": file_hash(__file__)})


if __name__ == "__main__":
    main()
