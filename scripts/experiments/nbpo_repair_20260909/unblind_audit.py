"""Join a previously sealed qualitative annotation set to its separately held key."""
import argparse
from collections import Counter
from pathlib import Path

from scripts.experiments.nbpo_repair_20260909.common import file_hash, read_jsonl, write_json, write_jsonl
from scripts.experiments.nbpo_repair_20260909.evaluate_responses import bootstrap_ratio, paired_delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--directory", type=Path, required=True)
    args = ap.parse_args()
    p = args.directory / "annotations.jsonl"
    if file_hash(p) != "1b7fcc422aee4f69a918cc4c637f05f54cd039c84fcc76d535e796c1dbd2696d":
        raise ValueError("Annotations differ from the blinded reviewer's sealed artifact")
    key = {(r["audit_id"], r["blind_label"]): r for r in read_jsonl(args.directory / "unblinding_key.jsonl")}
    annotations = read_jsonl(p)
    if len(key) != 300 or len(annotations) != 300 or set(key) != {(r["audit_id"], r["blind_label"]) for r in annotations}:
        raise ValueError("Must join exactly100 prompts x3 models")
    joined = [{**r, **key[(r["audit_id"], r["blind_label"])], "status": "ok"} for r in annotations]
    labels = sorted({r["model"] for r in joined})
    groups = {label: sorted([r for r in joined if r["model"] == label], key=lambda r: r["uid"]) for label in labels}
    if any(len(rows) != 100 for rows in groups.values()):
        raise ValueError("Expected100 permodel")
    metrics = ["refusal", "extreme_brevity", "enough_answer", "normal_substantive", "repetition_degeneration",
               "text_incomplete", "text_content_incomplete", "format_only_incomplete"]
    summary = {"source_annotations_sha256": file_hash(p), "unblinding_key_sha256": file_hash(args.directory / "unblinding_key.jsonl"),
        "method": "Previously model-blinded coding-agent-assisted qualitative review, not human or validated classifier",
        "limitations": ["Same100 prompts permodel; overlapping categories", "No earlyEOS causal evidence without rawterminals",
            "Prompt-only exclusions fixed before response adjudication", "Qualitative adequacy/error flags are not an independently verified factualaccuracy benchmark",
            "Pairedprompt CIs conditional on this one codingagent's labels, not adjudicator or trainingseed uncertainty"],
        "models": {}}
    for label, rows in groups.items():
        summary["models"][label] = {"n_prompts": len(rows), "counts": {k: sum(r[k] for r in rows) for k in metrics},
            "rates": {k: bootstrap_ratio([r[k] for r in rows]) for k in metrics},
            "finish_reason_counts": dict(Counter(r["source_finish_reason"] for r in rows)),
            "refusal_kinds": dict(Counter(r["refusal_kind"] for r in rows if r["refusal"]))}
        if label != "base":
            summary["models"][label]["paired_delta_vs_base"] = {k: paired_delta(rows, groups["base"], k) for k in metrics}
    write_jsonl(args.directory / "unblinded_annotations_v1.jsonl", joined)
    write_json(args.directory / "unblinded_summary_v1.json", summary)
    print({label: report["counts"] for label, report in summary["models"].items()})


if __name__ == "__main__":
    main()
