"""Materialize the same immutable pair DatasetDict for WBC and online MSE.

No model or reference forward is performed. The dataset manifest pins source
pair files and every saved Arrow shard; pass its SHA to both training configs.
"""
import argparse
import hashlib
import json
from pathlib import Path
from datasets import DatasetDict, load_dataset

from mnpo_scripts.nbpo_neural import validate_canonical_pair_dataset
from mnpo_scripts.precompute_provenance import write_precompute_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provenance", required=True,
                        help="JSON with immutable model/tokenizer/pool/teacher/solver hashes")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("Refusing to overwrite an existing dataset artifact")
    provenance = json.loads(Path(args.provenance).read_text())
    required = {"model_revision", "tokenizer_hash", "chat_template_hash",
                "train_pool_sha256", "dev_pool_sha256", "teacher_manifest_sha256"}
    if not required <= provenance.keys() or any(not provenance[k] for k in required):
        raise ValueError(f"Provenance must declare {sorted(required)}")
    dataset = DatasetDict()
    reports = {}
    for split, path in (("train", args.train), ("dev", args.dev)):
        dataset[split] = load_dataset("json", data_files=str(path), split="train")
        reports[split] = validate_canonical_pair_dataset(dataset[split])
        reports[split]["pair_file_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    dataset.save_to_disk(str(output))
    (output / "nbpo_dataset_provenance.json").write_text(
        json.dumps({"provenance": provenance, "splits": reports}, indent=2) + "\n")
    manifest_path, manifest_hash = write_precompute_manifest(str(output), splits=["train", "dev"])
    print(json.dumps({"dataset": str(output), "manifest": manifest_path,
                      "nbpo_expected_dataset_manifest_sha256": manifest_hash,
                      "reports": reports}, indent=2))


if __name__ == "__main__":
    main()
