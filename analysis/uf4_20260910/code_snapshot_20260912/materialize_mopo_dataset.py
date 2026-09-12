"""Materialize the MOPO rows as a training DatasetDict, with the pinned provenance.

The canonical path in run_mnpo bundles provenance pinning with a 28-pair
structural validator, which a one-row-per-candidate dataset can never satisfy.
That block is now split (see provenance/run_mnpo.py.before_mopo_split), so this
dataset declares nbpo_target_mode: mopo_rho and gets the same pinning every
other arm gets -- manifest hash, tokenizer and chat-template hashes, model
revision -- with validate_mopo_rho_dataset in place of the pair validator.

The checks below are re-established here as well as in the trainer, because a
dataset that is wrong is cheaper to catch at build time than at step 0 of a
four-GPU run: the chat-template hash carried by the rows is recomputed from the
tokenizer the training run will actually load, and the pool's model revision is
compared against the training revision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from datasets import DatasetDict, load_dataset

ROOT = Path("/work/uf4_20260910")
REQUIRED = ("prompt_input_ids", "prompt_attention_mask",
            "chosen_input_ids", "chosen_attention_mask", "chosen_labels",
            "rejected_input_ids", "rejected_attention_mask", "rejected_labels",
            "prompt", "chosen", "rejected", "prompt_id",
            "ronpo_target", "target_mode", "target_units",
            "nbpo_num_candidates", "solver_artifact_sha256")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def one_value(dataset, column, split):
    values = set(dataset.unique(column))
    if len(values) != 1:
        raise ValueError("%s: %s is not constant (%d values)" % (split, column, len(values)))
    return values.pop()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", default="mopo_v1")
    ap.add_argument("--out-name", default="mopo_v1")
    ap.add_argument("--base-model", default="/work/models/bases/Llama-3.1-8B-Instruct")
    ap.add_argument("--model-revision", required=True)
    args = ap.parse_args()

    target_dir = ROOT / "targets" / args.targets
    files = {s: str(target_dir / "pairs" / f"{s}.jsonl") for s in ("train", "dev")}
    out = ROOT / "datasets" / args.out_name
    if out.exists():
        raise SystemExit("%s already exists; remove it deliberately rather than overwriting" % out)

    dataset, reports = DatasetDict(), {}
    for split, path in files.items():
        d = load_dataset("json", data_files=path, split="train")
        missing = [c for c in REQUIRED if c not in d.column_names]
        if missing:
            raise ValueError("%s: row file lacks %s" % (split, missing))
        mode = one_value(d, "target_mode", split)
        units = one_value(d, "target_units", split)
        if (mode, units) != ("mopo_rho", "importance_ratio"):
            raise ValueError("%s: declares (%s, %s)" % (split, mode, units))
        n_cand = int(one_value(d, "nbpo_num_candidates", split))
        if d.num_rows % n_cand:
            raise ValueError("%s: %d rows is not a multiple of %d candidates"
                             % (split, d.num_rows, n_cand))
        prompts = len(set(d.unique("prompt_id")))
        if d.num_rows != prompts * n_cand:
            raise ValueError("%s: %d rows for %d prompts x %d candidates"
                             % (split, d.num_rows, prompts, n_cand))
        rho = d["ronpo_target"]
        mean = sum(rho) / len(rho)
        if abs(mean - 1.0) > 1e-6:
            raise ValueError("%s: mean rho %.9f is not 1; not an importance ratio" % (split, mean))
        reports[split] = {
            "rows": d.num_rows, "prompts": prompts, "candidates": n_cand,
            "target_mode": mode, "target_units": units,
            "rho_mean": mean, "rho_min": min(rho), "rho_max": max(rho),
            "model_revision": one_value(d, "model_revision", split),
            "chat_template_hash": one_value(d, "chat_template_hash", split),
            "solver_artifact_sha256": one_value(d, "solver_artifact_sha256", split),
            "row_file": path, "row_file_sha256": file_hash(path)}
        dataset[split] = d

    from transformers import AutoTokenizer
    from mnpo_scripts.precompute_provenance import (tokenizer_content_hashes,
                                                    write_precompute_manifest,
                                                    PRECOMPUTE_MANIFEST_FILENAME)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    hashes = tokenizer_content_hashes(tokenizer)
    for split, report in reports.items():
        if report["chat_template_hash"] != hashes["chat_template_hash"]:
            raise ValueError("%s: row chat template does not match the training tokenizer" % split)
        if str(report["model_revision"]) != str(args.model_revision):
            raise ValueError("%s: pool revision %s != training revision %s"
                             % (split, report["model_revision"], args.model_revision))
    if len({r["solver_artifact_sha256"] for r in reports.values()}) != 1:
        raise ValueError("train and dev name different solver artifacts")

    dataset.save_to_disk(str(out))

    provenance = {"provenance": {
        "tokenizer_hash": hashes["tokenizer_hash"],
        "chat_template_hash": hashes["chat_template_hash"],
        "model_revision": args.model_revision,
        "reduction": "sequence_sum",
        "label": ("MOPO importance weight rho(y) of Agnihotri et al. Eq. (3); one row per "
                  "(prompt, candidate), loss -rho(y) log pi(y) on the chosen side only"),
        "target_mode": "mopo_rho", "target_units": "importance_ratio",
        "tokens": ("harvested from the canonical pair file, so bit-identical to the tokens the "
                   "NBPO, fixed-reference, utilitarian and maxmin arms trained on"),
        "rejected_side": "carried for the collator; enters no term of the MOPO loss",
        "splits": reports,
        "builder_sha256": file_hash(Path(__file__))}}
    (out / "nbpo_dataset_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    # The manifest must be written LAST: it hashes every file in the directory
    # except itself, so writing it before the provenance sidecar leaves the
    # sidecar unlisted and verify_precompute_manifest rejects the dataset with
    # added=['nbpo_dataset_provenance.json']. That is exactly how the first
    # MOPO run failed.
    write_precompute_manifest(str(out))
    manifest_sha = file_hash(out / PRECOMPUTE_MANIFEST_FILENAME)

    # make_train_jobs reads these two, and hashes train/solver/solution.json.
    complete_path = target_dir / "complete.json"
    complete = json.loads(complete_path.read_text())
    complete["dataset_path"] = str(out)
    complete["dataset_manifest_sha256"] = manifest_sha
    complete_path.write_text(json.dumps(complete, indent=2) + "\n")

    solver_dir = target_dir / "train/solver"
    solver_dir.mkdir(parents=True, exist_ok=True)
    solution = solver_dir / "solution.json"
    if not solution.exists():
        # MOPO's solve writes rho.npz rather than a global solver solution; the
        # shared job generator hashes this path, so it is given a real artifact
        # that binds to the rho file instead of a placeholder.
        rho_path = target_dir / "train/rho.npz"
        solution.write_text(json.dumps({
            "aggregation": "mopo_constrained",
            "note": ("MOPO has no global weight vector: the dual is lambda over the secondary "
                     "objectives and the trained quantity is the per-candidate importance ratio "
                     "rho(y). This file exists so the shared job generator can hash the solver "
                     "artifact, and it binds to the rho file it was produced from."),
            "rho_path": str(rho_path), "rho_sha256": file_hash(rho_path),
            "declarations": complete["declarations"]}, indent=2) + "\n")

    print(json.dumps({"dataset": str(out), "manifest_sha256": manifest_sha,
                      "train_rows": reports["train"]["rows"],
                      "dev_rows": reports["dev"]["rows"],
                      "solver_solution": str(solution)}), flush=True)


if __name__ == "__main__":
    main()
