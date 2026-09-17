"""Write a parser-compatible, artifact-bound primary or disposable profile config."""
import argparse
import dataclasses
import json
from pathlib import Path
import yaml

from alignment import ModelArguments, DataArguments
from mnpo_scripts.mnpo_config import MNPOConfig
from mnpo_scripts.precompute_provenance import sha256_file_hex, verify_precompute_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model", default="/work/models/bases/Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-output", required=True, type=Path)
    parser.add_argument("--loss", required=True, choices=("nbpo", "nbpo_wbc"))
    parser.add_argument("--steps", type=int, default=1750)
    parser.add_argument("--world-size", type=int, choices=(2, 4), default=4)
    parser.add_argument("--profile", action="store_true",
                        help="Disposable 20-update runtime measurement; no eval/checkpoint selection")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "training_configs/nbpo/repair_20260909/train_common.yaml").read_text())
    manifest_path = args.dataset / "precompute_manifest.json"
    manifest_hash = sha256_file_hex(str(manifest_path))
    verify_precompute_manifest(str(args.dataset), manifest_hash)
    provenance = json.loads((args.dataset / "nbpo_dataset_provenance.json").read_text())
    solver_hashes = provenance["splits"]["train"]["solver_artifact_sha256"]
    if len(solver_hashes) != 1:
        raise ValueError("One primary train split must use exactly one global-lambda solver artifact")
    config.update({
        "model_name_or_path": args.model, "model_revision": provenance["provenance"]["model_revision"],
        "dataset_mixer": {str(args.dataset): 1.0}, "output_dir": args.output_dir,
        "run_name": Path(args.output_dir).name, "loss_type": args.loss,
        "nbpo_expected_dataset_manifest_sha256": manifest_hash,
        "nbpo_expected_solver_artifact_sha256": solver_hashes[0],
        "nbpo_online_reference": args.loss == "nbpo",
        "nbpo_eval_online_reference": True,
        "nbpo_verify_reference_initialization": args.loss == "nbpo",
        "nbpo_reference_model_path": args.model,
        "gradient_accumulation_steps": 32 // args.world_size,
        "deepspeed": str(root / "training_configs/nbpo/repair_20260909/deepspeed_zero2.json"),
        "max_steps": args.steps,
    })
    if args.profile:
        config.update({"eval_strategy": "no", "save_strategy": "no",
                       "nbpo_eval_online_reference": False, "nbpo_profile_updates": 20})
    if config["max_steps"] <= 0:
        raise ValueError("Training horizon must be fixed and positive")
    supported = {field.name for cls in (ModelArguments, DataArguments, MNPOConfig)
                 for field in dataclasses.fields(cls)}
    if set(config) - supported:
        raise ValueError(f"Unsupported actual parser keys: {sorted(set(config) - supported)}")
    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    with args.config_output.open("x") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    print(json.dumps({"config": str(args.config_output), "sha256": sha256_file_hex(str(args.config_output)),
                      "loss": args.loss, "updates": config["max_steps"],
                      "profile_stop_after": 20 if args.profile else None,
                      "global_batch_pairs": args.world_size * config["gradient_accumulation_steps"],
                      "profile_only": args.profile}))


if __name__ == "__main__":
    main()
