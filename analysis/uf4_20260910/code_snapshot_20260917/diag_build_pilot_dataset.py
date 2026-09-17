"""Build the projection pilot's dataset: the frozen 200 train and 200 dev prompts.

Rows are the pairs already on disk for nash_v1 -- no target is re-solved and no
pair is dropped, so all 28 unordered pairs of every prompt are present and the
trainer's structural validator sees the same contract as the full run. Train
rows are written in panel order with a prompt's 28 pairs consecutive, which is
what lets one optimizer step cover exactly one prompt's complete pair graph.

The dataset is materialized through mnpo_scripts.prepare_nbpo_dataset, the same
path the campaign datasets came from, so the precompute manifest and provenance
are produced by the validated writer rather than by hand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
DIAG = ROOT / "analysis/diag_20260914"


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def filter_pairs(src, keep_order, dest):
    """Write only the wanted prompts, grouped and ordered by keep_order."""
    rank = {pid: k for k, pid in enumerate(keep_order)}
    rows = {}
    total = 0
    with open(src) as stream:
        for line in stream:
            row = json.loads(line)
            pid = row["prompt_id"]
            if pid in rank:
                rows.setdefault(pid, []).append(line)
            total += 1
    ordered, counts = [], {}
    for pid in keep_order:
        got = rows.get(pid, [])
        counts[len(got)] = counts.get(len(got), 0) + 1
        ordered.extend(got)
    tmp = Path(str(dest) + ".tmp")
    with tmp.open("w") as stream:
        stream.writelines(ordered)
    tmp.replace(dest)
    return {"source_rows": total, "kept_rows": len(ordered),
            "prompts": len(keep_order), "pairs_per_prompt_histogram": counts}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", default="nash_v1")
    ap.add_argument("--dataset-name", default="diag_pilot200")
    ap.add_argument("--model", default="/work/models/bases/Llama-3.1-8B-Instruct")
    args = ap.parse_args()

    out_dir = DIAG / "projection"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_panel = json.loads((DIAG / "panel_train200.json").read_text())
    dev_panel = json.loads((DIAG / "panel_dev200.json").read_text())
    target_dir = ROOT / "targets" / args.targets

    train_stats = filter_pairs(target_dir / "pairs/train.jsonl",
                               train_panel["panel_prompt_ids"],
                               out_dir / "pairs_train200.jsonl")
    dev_stats = filter_pairs(target_dir / "pairs/dev.jsonl",
                             dev_panel["panel_prompt_ids"],
                             out_dir / "pairs_dev200.jsonl")

    provenance = target_dir / "dataset_provenance.repaired.json"
    if not provenance.exists():
        provenance = target_dir / "dataset_provenance.json"
    dataset_out = ROOT / "datasets" / args.dataset_name
    env = dict(os.environ, HF_DATASETS_CACHE=str(out_dir / "arrow_cache"),
               PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(
        [sys.executable, "-m", "mnpo_scripts.prepare_nbpo_dataset",
         "--train", str(out_dir / "pairs_train200.jsonl"),
         "--dev", str(out_dir / "pairs_dev200.jsonl"),
         "--output", str(dataset_out), "--provenance", str(provenance)],
        env=env, capture_output=True, text=True)
    if completed.returncode:
        (out_dir / "dataset_build_failure.json").write_text(json.dumps(
            {"returncode": completed.returncode, "stdout": completed.stdout[-4000:],
             "stderr": completed.stderr[-4000:]}, indent=2) + "\n")
        raise SystemExit("prepare_nbpo_dataset failed; diagnostic preserved")

    manifest = file_hash(dataset_out / "precompute_manifest.json")
    summary = {"dataset": str(dataset_out), "manifest_sha256": manifest,
               "provenance_used": str(provenance),
               "train": train_stats, "dev": dev_stats,
               "train_panel_sha256": train_panel["panel_sha256"][:16],
               "dev_panel_sha256": dev_panel["panel_sha256"][:16],
               "rows_per_update_plan": {"gpus": 4, "per_device": 1,
                                        "grad_accum": 7, "rows_per_update": 28,
                                        "note": "one prompt's complete 28-pair graph per step"}}
    (out_dir / "dataset_build.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
