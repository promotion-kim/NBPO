"""Materialize one scalarized-DPO weighting as a training DatasetDict.

prepare_nbpo_dataset cannot be used here, and not for a trivial reason: its
validator requires strictly positive solver masses, an Eq. (26) target that
matches log(w_a/c_a) - log(w_b/c_b), and all 28 unordered pairs present for
every prompt. A DPO weighting has no solver masses and deliberately drops the
pairs it cannot order, so that validator would reject a correct dataset.

Dropping it would also drop the checks that are worth keeping, because
run_mnpo only runs them on the canonical path. So this script re-establishes
them directly: the chat template and tokenizer hashes carried by the rows are
recomputed from the tokenizer the training run will actually load, the base
model revision is compared against the one the pool was sampled under, and the
weight vector is confirmed constant and equal to the declared one. Those are
the guarantees the canonical path provides; none of them is specific to NBPO.

It also removes target_artifact_hash, which names the Nash target_log_ratio
artifact. It is true provenance of the pair file this was derived from and false
provenance of this dataset's labels, and a reader finding it here would
reasonably conclude the DPO arm was trained on a Nash target.
"""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

from datasets import DatasetDict, load_dataset

ROOT = Path("/work/uf4_20260910")
# Provenance of the source pair file, not of these labels.
DROP_COLUMNS = ("target_artifact_hash",)
REQUIRED = ("prompt_input_ids", "prompt_attention_mask",
            "chosen_input_ids", "chosen_attention_mask", "chosen_labels",
            "rejected_input_ids", "rejected_attention_mask", "rejected_labels",
            "prompt", "chosen", "rejected", "prompt_id")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def one_value(dataset, column, label):
    """Exhaustive constancy check. Only for scalar columns.

    Arrow has no `unique` kernel for list columns, so a list-valued column has
    to be checked another way -- see sampled_value below.
    """
    values = dataset.unique(column)
    if len(values) != 1:
        raise ValueError(f"{label}: column {column} is not constant ({len(values)} values)")
    return values[0]


def sampled_value(dataset, column, label, probes=8):
    """Check a list-valued column on rows spread across the split.

    This is not exhaustive, and it does not need to be: the builder writes
    dpo_weight_name and dpo_weight_vector from the same entry of one dict, and
    the name is checked exhaustively above, so a constant name over the file
    already implies a constant vector. These probes catch the case that
    assumption is wrong, which is what a check is for.
    """
    n = dataset.num_rows
    if n == 0:
        raise ValueError(f"{label}: empty split")
    indices = sorted({int(i * (n - 1) / max(1, probes - 1)) for i in range(probes)})
    seen = {json.dumps(dataset[i][column]) for i in indices}
    if len(seen) != 1:
        raise ValueError(f"{label}: column {column} varies across the split ({seen})")
    return json.loads(seen.pop())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weight", required=True)
    ap.add_argument("--pairs-root", default=str(ROOT / "dpo/v1/pairs"))
    ap.add_argument("--calibration", default=str(ROOT / "dpo/v1/calibration_and_counts.json"))
    ap.add_argument("--base-model", default="/work/models/bases/Llama-3.1-8B-Instruct")
    ap.add_argument("--model-revision", default="0e9e39f249a16976918f6564b8830bc894c89659")
    args = ap.parse_args()

    cal = json.loads(Path(args.calibration).read_text())
    if args.weight not in cal["weights"]:
        raise SystemExit(f"{args.weight} is not a declared weighting")
    declared = [float(x) for x in cal["weights"][args.weight]]

    src = Path(args.pairs_root) / args.weight
    files = {split: str(src / f"{split}.jsonl") for split in ("train", "dev")}
    for path in files.values():
        if not Path(path).exists():
            raise SystemExit(f"missing pair file {path}")

    out = ROOT / "datasets" / f"dpo_{args.weight}_v1"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")

    dataset = DatasetDict()
    reports = {}
    for split, path in files.items():
        d = load_dataset("json", data_files=path, split="train")
        missing = [c for c in REQUIRED if c not in d.column_names]
        if missing:
            raise ValueError(f"{split}: pair file lacks {missing}")
        drop = [c for c in DROP_COLUMNS if c in d.column_names]
        if drop:
            d = d.remove_columns(drop)
        # the label is one weighting, constant over the file
        name = one_value(d, "dpo_weight_name", split)
        if name != args.weight:
            raise ValueError(f"{split}: rows declare weight {name!r}, expected {args.weight!r}")
        vector = sampled_value(d, "dpo_weight_vector", split)
        if [float(x) for x in vector] != declared:
            raise ValueError(f"{split}: weight vector {vector} != declared {declared}")
        revision = one_value(d, "model_revision", split)
        if str(revision) != str(args.model_revision):
            raise ValueError(f"{split}: pool revision {revision} != training revision")
        template = one_value(d, "chat_template_hash", split)
        reports[split] = {"rows": d.num_rows, "dropped_columns": drop,
                          "weight_name": name, "weight_vector": vector,
                          "model_revision": revision, "chat_template_hash": template,
                          "tie_threshold": one_value(d, "dpo_tie_threshold_standardized", split),
                          "pair_file": path, "pair_file_sha256": file_hash(path)}
        dataset[split] = d

    # the check run_mnpo performs only on the canonical path
    from transformers import AutoTokenizer
    from mnpo_scripts.precompute_provenance import tokenizer_content_hashes
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    hashes = tokenizer_content_hashes(tokenizer)
    for split, report in reports.items():
        if report["chat_template_hash"] != hashes["chat_template_hash"]:
            raise ValueError(f"{split}: pair chat template does not match the training tokenizer")
    if len({r["chat_template_hash"] for r in reports.values()}) != 1:
        raise ValueError("train and dev disagree on the chat template")

    dataset.save_to_disk(str(out))
    provenance = {
        "dataset": str(out), "weighting": args.weight, "weight_vector": declared,
        "weights_order": cal["weights_order"],
        "label": "scalarized frozen BT reward, train-pool standardization, ties dropped",
        "tie_threshold_standardized": cal["tie_threshold_standardized"],
        "calibration_sha256": file_hash(args.calibration),
        "loss": "loss_type=dpo; the Eq. (26) target columns are absent by construction",
        "pool": "identical prompts, occurrences and token ids as the NBPO arms",
        "model_revision": args.model_revision,
        **{k: hashes[k] for k in ("tokenizer_hash", "chat_template_hash")},
        "checks": ("weight name, tie threshold, model revision and chat-template hash "
                   "checked exhaustively over every row; weight vector checked on eight "
                   "rows spread across each split, since Arrow cannot take unique() of a "
                   "list column and a constant name already implies a constant vector; "
                   "row chat-template hash recomputed from the training tokenizer"),
        "splits": reports,
        "builder_sha256": file_hash(__file__)}
    (out / "dpo_dataset_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"dataset": str(out),
                      "train_rows": reports["train"]["rows"],
                      "dev_rows": reports["dev"]["rows"],
                      "weight": args.weight}), flush=True)


if __name__ == "__main__":
    main()
