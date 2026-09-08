#!/usr/bin/env python3
"""Pick the winner of one tuning stage, on VALIDATION only, and record why.

Selection reads a single field -- validation ``normalized_mse_var`` -- and the
test half is not consulted, not printed and not ranked. That is enforced here
rather than left to discipline: the test metrics are copied into the record for
later reporting, but the ``argmin`` never sees them.

The record is append-only across stages (eta, then learning rate, then steps),
so the sweep can be read back as what was tried, in what order, and what each
stage selected -- not just the final configuration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gates", nargs="+", type=Path, required=True,
                    help="gate.json files, one per arm of THIS stage")
    ap.add_argument("--stage", required=True, choices=["eta", "learning_rate", "steps"])
    ap.add_argument("--record", type=Path, required=True)
    args = ap.parse_args()

    arms = []
    for g in args.gates:
        d = json.loads(g.read_text())
        v = d["splits"]["validation"]["metrics"]
        arms.append({
            "label": d["label"], "eta": d["eta"], "gate_file": str(g),
            "validation": v,
            "validation_gate": d["splits"]["validation"]["gate"],
            "test": d["splits"]["test"]["metrics"],
            "test_gate": d["splits"]["test"]["gate"],
        })

    scorable = [a for a in arms if a["validation"]["normalized_mse_var"] is not None]
    if not scorable:
        raise SystemExit("no arm produced a validation normalized_mse_var")
    winner = min(scorable, key=lambda a: a["validation"]["normalized_mse_var"])

    record = json.loads(args.record.read_text()) if args.record.exists() else {"stages": []}
    record["stages"].append({
        "stage": args.stage,
        "selection_metric": "validation normalized_mse_var (lower is better)",
        "selected": winner["label"],
        "arms": [{"label": a["label"],
                  "validation_normalized_mse_var": a["validation"]["normalized_mse_var"],
                  "validation_sign_agreement": a["validation"]["sign_agreement"],
                  "validation_pearson": a["validation"]["pearson"],
                  "validation_spearman": a["validation"]["spearman"],
                  "validation_prediction_rms": a["validation"]["prediction_rms"],
                  "validation_target_rms": a["validation"]["target_rms"]}
                 for a in arms],
        "test_metrics_carried_but_not_used_for_selection": {
            a["label"]: a["test"] for a in arms},
    })
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2) + "\n")

    print(f"stage={args.stage}  selected={winner['label']}")
    for a in sorted(arms, key=lambda a: a["validation"]["normalized_mse_var"]
                    if a["validation"]["normalized_mse_var"] is not None else 9e9):
        v = a["validation"]
        mark = " <-- selected" if a["label"] == winner["label"] else ""
        print(f"  {a['label']:>10s}  nMSE(var) {v['normalized_mse_var']:.4f}  "
              f"sign {v['sign_agreement'] if v['sign_agreement'] is None else round(v['sign_agreement'],4)}  "
              f"pearson {v['pearson'] if v['pearson'] is None else round(v['pearson'],4)}  "
              f"pred_rms {v['prediction_rms']:.4f}  target_rms {v['target_rms']:.4f}{mark}")


if __name__ == "__main__":
    main()
