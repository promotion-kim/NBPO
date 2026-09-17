"""Turn a materialized scalarized-DPO weighting into a training config and job.

The recipe is the same resolved config the NBPO arms run: same base model and
revision, same learning rate and schedule, same optimizer, same global batch,
same sequence limits, same DeepSpeed plan, same 1,250 updates. Only three groups
of keys differ, and each has a reason:

  * loss_type becomes dpo with dpo_beta 0.01. With logp_reduction "sum" the
    log-ratios are sequence sums, so the beta that pairs with them is small;
    0.01 is the library default for this trainer and is NOT tuned here. Tuning
    trials for this baseline are therefore zero, which is what the cost table
    must report.
  * the Eq. (26) target keys are removed. nbpo_target_mode in particular must
    not stay at canonical_logratio: that flag is what makes run_mnpo demand a
    solver certificate and all 28 pairs per prompt, neither of which a DPO
    weighting has.
  * the reference keys are KEPT. DPO needs a frozen reference, and computing it
    online from the base checkpoint is the same mechanism, tokenizer and chat
    template the other arms use. Substituting a cached reference here would make
    the comparison a comparison of reference handling as well.

The realized diff against the base recipe is written into the provenance record,
so a silent recipe change between arms is visible rather than implied.
"""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

import yaml

ROOT = Path("/work/uf4_20260910")
BASE_CONFIG = Path("/work/nbpo_repair_20260909/configs/mse_short_primary_v1.yaml")
# Keys that describe the Eq. (26) teacher. A DPO run has no such target.
REMOVE = ("nbpo_target_mode", "nbpo_target_column", "nbpo_target_units",
          "nbpo_eta_already_included", "nbpo_require_immutable_tokens",
          "nbpo_num_candidates", "nbpo_expected_dataset_manifest_sha256",
          "nbpo_expected_solver_artifact_sha256")
TRAIN_ENV = {"PYTHONPATH": "/work/nbpo_repair_20260909/deps_train:/work/nbpo_repair_20260909/code",
             "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "4",
             "MNPO_DISABLE_APEX": "1", "HF_HUB_OFFLINE": "1",
             "TOKENIZERS_PARALLELISM": "false", "WANDB_MODE": "disabled",
             "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weight", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=1250)
    ap.add_argument("--eval-steps", type=int, default=250)
    ap.add_argument("--dpo-beta", type=float, default=0.01)
    ap.add_argument("--priority", type=int, required=True)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--smoke", action="store_true",
                    help="two updates on one card, to prove the config and dataset load")
    args = ap.parse_args()

    base = yaml.safe_load(BASE_CONFIG.read_text())
    config = dict(base)
    removed = [k for k in REMOVE if k in config]
    for key in removed:
        config.pop(key)

    dataset = ROOT / "datasets" / f"dpo_{args.weight}_v1"
    arm = ("smoke_dpo_%s" % args.weight) if args.smoke else \
          ("dpo_%s_mse_s%d" % (args.weight, args.seed))
    config.update({
        "loss_type": "dpo",
        "dpo_beta": args.dpo_beta,
        "max_steps": 2 if args.smoke else args.max_steps,
        "seed": args.seed, "data_seed": args.seed,
        "dataset_mixer": {str(dataset): 1.0},
        "output_dir": str(ROOT / "arms" / arm),
        "run_name": arm,
    })
    if args.smoke:
        config.update({"eval_strategy": "no", "save_strategy": "no",
                       "logging_steps": 1, "warmup_ratio": 0.0})
    else:
        config.update({"eval_steps": args.eval_steps, "save_steps": args.max_steps,
                       "eval_strategy": "steps", "save_strategy": "steps"})

    config_path = ROOT / "configs" / f"{arm}.yaml"
    body = yaml.safe_dump(config, sort_keys=True, default_flow_style=False)
    if config_path.exists() and config_path.read_text() != body:
        raise SystemExit(f"refusing to overwrite {config_path} with different content")
    config_path.write_text(body)

    spec = {
        "job_id": f"uf4_train_{arm}", "priority": args.priority, "gpus": args.gpus,
        "cwd": "/work/nbpo_repair_20260909/code", "env": TRAIN_ENV,
        "depends_on": [f"uf4_materialize_dpo_{args.weight}"],
        "command": ["python3", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                    f"--nproc_per_node={args.gpus}", "-m", "mnpo_scripts.run_mnpo",
                    str(config_path)],
        "timeout_s": 7200 if args.smoke else 86400,
        "artifacts": ([str(ROOT / "arms" / arm / "runtime_rank0.jsonl")] if args.smoke
                      else [str(ROOT / "arms" / arm / "config.json")]),
        "config_sha256": file_hash(config_path),
    }
    queue_path = ROOT / "jobs" / "queue" / f"{args.priority}_train_{arm}.json"
    payload = json.dumps(spec, indent=2) + "\n"
    if queue_path.exists() and queue_path.read_text() != payload:
        raise SystemExit(f"refusing to change existing spec {queue_path}")
    queue_path.write_text(payload)

    diff = {k: {"base": base.get(k, "<absent>"), "dpo": config.get(k, "<absent>")}
            for k in sorted(set(base) | set(config))
            if base.get(k, "<absent>") != config.get(k, "<absent>")}
    record = {"arm": arm, "weighting": args.weight, "seed": args.seed,
              "dpo_beta": args.dpo_beta,
              "dpo_beta_note": ("library default for this trainer, paired with sequence-sum "
                                "log-ratios; not tuned, so tuning trials for this baseline "
                                "are zero"),
              "dataset": str(dataset), "config_path": str(config_path),
              "config_sha256": file_hash(config_path),
              "base_config": str(BASE_CONFIG), "base_config_sha256": file_hash(BASE_CONFIG),
              "removed_keys": removed, "recipe_diff_vs_resolved_base": diff,
              "queue_spec": str(queue_path), "smoke": args.smoke}
    out = ROOT / "provenance" / f"train_job_{arm}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"queued": spec["job_id"], "gpus": args.gpus,
                      "changed_keys": sorted(diff), "removed": removed}), flush=True)


if __name__ == "__main__":
    main()
