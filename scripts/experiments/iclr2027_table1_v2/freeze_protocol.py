#!/usr/bin/env python3
"""Freeze the selected judge protocol before the holdout audit.

Section E requires the protocol text, model revision, tokenizer revision,
candidate-token mapping, entropy threshold, adjudication rule and hashes to be
fixed *before* the 200-prompt holdout is run, so that nothing can be tuned after
seeing it. This copies the winning candidate to `protocol.yaml`, records
everything that determines its behaviour in `protocol_manifest.json`, and states
in the manifest which calibration evidence selected it.

The manifest deliberately also records what the protocol does **not** clear, so
a later reader does not have to reconstruct which gates were open at freeze
time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

GATES = {
    "clear_control_directional_accuracy_min": ("directional_accuracy_clear", 0.90, "min"),
    "identical_confident_tie_accuracy_min": ("identical_confident_tie_accuracy", 0.90, "min"),
    "abs_signed_position_bias_max": ("position_bias", 0.03, "absmax"),
    "confident_decisive_swap_consistency_min": ("confident_pair_swap_consistency", 0.85, "min"),
    "split_half_spearman_min": ("split_half_spearman", 0.75, "min"),
}


def gate_report(per_objective: dict) -> dict:
    out = {}
    for name, (key, thr, mode) in GATES.items():
        vals = {o: v[key] for o, v in per_objective.items() if v.get(key) is not None}
        if not vals:
            out[name] = {"threshold": thr, "status": "unmeasured"}
            continue
        if mode == "min":
            worst = min(vals, key=lambda o: vals[o])
            ok = vals[worst] >= thr
        else:
            worst = max(vals, key=lambda o: abs(vals[o]))
            ok = abs(vals[worst]) <= thr
        out[name] = {"threshold": thr, "worst_objective": worst,
                     "value": round(vals[worst], 6), "pass": bool(ok)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", type=Path, required=True,
                    help="calibration_results.json from analyze_calibration")
    ap.add_argument("--candidates-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--judge-model-path", required=True)
    ap.add_argument("--judge-revision-fingerprint", type=Path, default=None)
    ap.add_argument("--select", default=None,
                    help="override the ranked winner; recorded with a reason")
    ap.add_argument("--select-reason", default=None)
    ap.add_argument("--git-commit", default=None)
    args = ap.parse_args()

    calib = json.loads(args.calibration.read_text())
    chosen = args.select or calib["selected"]
    if args.select and not args.select_reason:
        raise SystemExit("--select overrides the ranking; --select-reason is required "
                         "so the deviation is on the record")
    run = calib["runs"].get(chosen)
    if run is None:
        raise SystemExit(f"{chosen!r} is not in the calibration results")

    name_map = {k: v.get("manifest", {}).get("protocol_name") for k, v in calib["runs"].items()}
    protocol_name = name_map.get(chosen) or chosen
    src = args.candidates_dir / f"{protocol_name}.yaml"
    if not src.exists():
        raise SystemExit(f"selected protocol file {src} does not exist")

    commit = args.git_commit
    if commit is None:
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                             stderr=subprocess.DEVNULL).strip()
        except Exception:
            raise SystemExit("pass --git-commit; a frozen protocol without its source "
                             "commit cannot be re-derived")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dst = args.out_dir / "protocol.yaml"
    shutil.copyfile(src, dst)

    gates = gate_report(run["per_objective"])
    open_gates = [k for k, v in gates.items() if v.get("pass") is False]
    manifest = {
        "frozen_at_git_commit": commit,
        "selected_protocol": protocol_name,
        "selected_by": ("lexicographic ranking over calibration and reliability metrics "
                        "only" if not args.select else "operator override"),
        "override_reason": args.select_reason,
        "ranking": calib["ranking"],
        "selection_rule": calib["selection_rule"],
        "selection_inputs": calib["selection_inputs"],
        "protocol_file": {"path": str(dst),
                          "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()},
        "protocol_sha256": run.get("manifest", {}).get("protocol_sha256"),
        "rubric_sha256": run.get("manifest", {}).get("rubric_sha256"),
        "labels": run.get("manifest", {}).get("labels"),
        "label_mapping_note": run.get("manifest", {}).get("label_mapping_note"),
        "tokenization_verification": run.get("manifest", {}).get("tokenization_verification"),
        "entropy_threshold": run.get("manifest", {}).get("entropy_threshold"),
        "order_gap_threshold": run.get("manifest", {}).get("order_gap_threshold"),
        "max_templates_when_unstable": run.get("manifest", {}).get("max_templates_when_unstable"),
        "judge_model_path": args.judge_model_path,
        "calibration_source": str(args.calibration),
        "calibration_gate_status_on_development_controls": gates,
        "gates_open_at_freeze": open_gates,
        "gate_note": ("these are the Section F thresholds evaluated on the DEVELOPMENT "
                      "controls, recorded so a reader knows what was already failing "
                      "before the holdout ran. The holdout audit is the binding "
                      "evaluation and is a separate, later run on prompts this protocol "
                      "has never seen."),
        "frozen_fields": ["protocol text", "model revision", "tokenizer revision",
                          "candidate-token mapping", "entropy threshold",
                          "adjudication rule", "template set", "decoding"],
    }
    if args.judge_revision_fingerprint and args.judge_revision_fingerprint.exists():
        manifest["judge_revision_fingerprint"] = json.loads(
            args.judge_revision_fingerprint.read_text())

    (args.out_dir / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"selected": protocol_name, "ranking": calib["ranking"],
                      "gates_open_at_freeze": open_gates}, indent=2))


if __name__ == "__main__":
    main()
