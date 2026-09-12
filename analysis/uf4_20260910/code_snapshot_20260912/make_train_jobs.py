"""Turn a finished UF-4 target set into a training config and a queued job.

Runs as a queue job itself, so the moment a solve finishes its policy arm is
registered without anyone typing a hash. The recipe is the resolved config that
actually worked on the SafeRLHF panel; only the dataset, the horizon, the run
name and the two artifact hashes differ, and the diff is written into the job
record so a silent recipe change is visible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
BASE_CONFIG = Path("/work/nbpo_repair_20260909/configs/mse_short_primary_v1.yaml")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=1250)
    ap.add_argument("--priority", type=int, default=60)
    ap.add_argument("--loss-type", default="nbpo",
                    help="Opt-in only. Anything but nbpo also rewrites nbpo_target_mode, so an "
                         "arm whose dataset is not the canonical pair format gets its own "
                         "structural validator instead of the pair one. Every other recipe "
                         "anchor is left identical, which is what makes the arms matched.")
    ap.add_argument("--target-mode", default=None,
                    help="Defaults to canonical_logratio for nbpo and mopo_rho for mopo.")
    ap.add_argument("--eval-steps", type=int, default=250,
                    help="Diagnostic dev-eval cadence. Must be identical across arms "
                         "that are compared with each other.")
    args = ap.parse_args()

    target_dir = ROOT / "targets" / args.targets
    complete = json.loads((target_dir / "complete.json").read_text())
    dataset = complete["dataset_path"]
    manifest = complete["dataset_manifest_sha256"]
    solver = file_hash(target_dir / "train/solver/solution.json")
    if not dataset or not manifest:
        raise ValueError(f"{args.targets} has no materialized dataset")

    base = BASE_CONFIG.read_text()
    target_mode = args.target_mode or ("mopo_rho" if args.loss_type == "mopo"
                                       else "canonical_logratio")
    replacements = {
        "/work/nbpo_repair_20260909/datasets/nash_repair_v2: 1.0": f"{dataset}: 1.0",
        "output_dir: /work/nbpo_repair_20260909/arms/mse_short_primary_v1":
            f"output_dir: {ROOT}/arms/{args.arm}",
        "run_name: mse_short_primary_v1": f"run_name: {args.arm}",
        "nbpo_expected_dataset_manifest_sha256: 856ba968818f4672eace40d2794a1889a99ebd7e866da0531e64d57e01564fac":
            f"nbpo_expected_dataset_manifest_sha256: {manifest}",
        "nbpo_expected_solver_artifact_sha256: 4fd522a7891fa7ba4ab3bcfcd82f0d72a5b50de23a8b9239a8e480b74588afc8":
            f"nbpo_expected_solver_artifact_sha256: {solver}",
        "max_steps: 250": f"max_steps: {args.max_steps}",
        # Newline-anchored: "seed: 42" is a substring of "data_seed: 42", so an
        # unanchored replacement rewrites both and then loses its own anchor.
        # At seed 42 that was invisible because the rewrite was a no-op.
        "\nseed: 42": f"\nseed: {args.seed}",
        "\ndata_seed: 42": f"\ndata_seed: {args.seed}",
        "save_steps: 250": f"save_steps: {args.max_steps}",
        # Diagnostic eval only: load_best_model_at_end is false and save_steps
        # equals max_steps, so nothing selects on it. Measured cost is 15 min 45 s
        # per pass on the 28,000-pair dev set at eval batch 1, so eval_steps=50 on
        # a 1250-update run spends 82% of wall time and 26 GPU-hours on logging.
        # 250 restores the five evaluations the validated 250-update recipe ran.
        "eval_steps: 50": f"eval_steps: {args.eval_steps}",
    }
    if args.loss_type != "nbpo":
        replacements["loss_type: nbpo"] = f"loss_type: {args.loss_type}"
    if target_mode != "canonical_logratio":
        replacements["nbpo_target_mode: canonical_logratio"] = f"nbpo_target_mode: {target_mode}"
    config = base
    applied = []
    for old, new in replacements.items():
        if old not in config:
            raise ValueError(f"Recipe anchor missing from the resolved base config: {old!r}")
        config = config.replace(old, new)
        applied.append({"from": old, "to": new})
    config_path = ROOT / "configs" / f"{args.arm}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists() and config_path.read_text() != config:
        raise ValueError(f"Refusing to overwrite {config_path} with different content")
    config_path.write_text(config)

    spec = {
        "job_id": f"uf4_train_{args.arm}",
        "priority": args.priority,
        "gpus": 4,
        "cwd": "/work/nbpo_repair_20260909/code",
        "env": {"PYTHONPATH": "/work/nbpo_repair_20260909/deps_train:/work/nbpo_repair_20260909/code",
                "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "4",
                "MNPO_DISABLE_APEX": "1", "HF_HUB_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false", "WANDB_MODE": "disabled",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
        "command": ["python3", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                    "--nproc_per_node=4", "-m", "mnpo_scripts.run_mnpo", str(config_path)],
        "timeout_s": 86400,
        "artifacts": [f"{ROOT}/arms/{args.arm}/config.json"],
        # The spec names the config by path, so without this a changed recipe
        # leaves the spec byte-identical and a queue keyed on spec bytes would
        # never notice that this is a different run.
        "config_sha256": file_hash(config_path),
        "dataset_manifest_sha256": manifest,
        "solver_artifact_sha256": solver,
    }
    queue_path = ROOT / "jobs" / "queue" / f"{args.priority}_train_{args.arm}.json"
    queue_path.write_text(json.dumps(spec, indent=2) + "\n")

    record = {"arm": args.arm, "targets": args.targets, "dataset": dataset,
              "loss_type": args.loss_type, "nbpo_target_mode": target_mode,
              "dataset_manifest_sha256": manifest, "solver_artifact_sha256": solver,
              "config_path": str(config_path), "config_sha256": file_hash(config_path),
              "base_config": str(BASE_CONFIG), "base_config_sha256": file_hash(BASE_CONFIG),
              "recipe_diff_vs_resolved_base": applied,
              "horizon_note": ("1250 updates at 32 pairs is the planned exposure of about four "
                               "pairs per prompt over 10,000 prompts. It is the plan's starting "
                               "horizon, not a horizon shown to be optimal on this data."),
              "queue_spec": str(queue_path)}
    out = ROOT / "provenance" / f"train_job_{args.arm}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"queued": spec["job_id"], "config": str(config_path),
                      "dataset_manifest": manifest[:16], "solver": solver[:16]}), flush=True)


if __name__ == "__main__":
    main()
