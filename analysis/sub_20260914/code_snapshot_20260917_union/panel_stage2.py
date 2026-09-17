"""Run one union panel from its judgments to its queued training jobs.

Steps, each idempotent and each recorded:

  1 score       order-balanced PSC blocks for the panel's K objectives
  2 probe       per-prompt feasibility of all three per-prompt rules -> <tag>p
  3 probe-solve the three per-prompt rules and the global rule, no dataset
  4 represent   drop any prompt whose optimal mass is below the representable
                threshold in ANY arm, from EVERY arm -> <tag>r
  5 solve       the same solvers again on the common representable split, this
                time materializing each arm's all-pair dataset
  6 baselines   objective-averaged soft labels (scalarized DPO, mean-preference
                INPO) and the PROSPER fitting pairs
  7 queue       one training spec per arm

Nothing here invents a comparison: every arm reads the same prompts, the same
pool occurrences and the same judgments, and differs only in the compromise rule
or loss it implements. An arm whose baseline has no implementation is left out
rather than replaced by a surrogate.
"""
from __future__ import annotations

import argparse, glob, hashlib, json, math, os, subprocess, sys, time
from pathlib import Path

import numpy as np

UF = Path("/work/uf4_20260910")
SUB = Path("/work/sub_20260914")
CODE = SUB / "code"
DEPS = "/work/nbpo_repair_20260909"
BASE_MODEL = "/work/models/bases/Qwen2.5-7B-Instruct"
BASE_REV = "a09a35458c702b33eeacc393d103063234e8bc28"
MIN_MASS = 1e-10
POOL = 8

SOLVE_ENV = {"PYTHONPATH": "%s/code:%s:%s" % (DEPS, UF / "code", CODE),
             "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
             "HF_HUB_OFFLINE": "1"}
WRITE_ENV = {"PYTHONPATH": "%s/deps_train:%s/code:%s:%s" % (DEPS, DEPS, UF / "code", CODE),
             "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
             "HF_HUB_OFFLINE": "1"}


def run(cmd, env_extra, label, log):
    env = dict(os.environ)
    env.update({k: str(v) for k, v in env_extra.items()})
    t0 = time.monotonic()
    with open(log, "a") as sink:
        sink.write("\n===== %s =====\n%s\n" % (label, " ".join(cmd)))
        sink.flush()
        rc = subprocess.call(cmd, env=env, cwd=str(CODE), stdout=sink,
                             stderr=subprocess.STDOUT)
    took = round(time.monotonic() - t0, 1)
    print(json.dumps({"step": label, "returncode": rc, "seconds": took}), flush=True)
    if rc != 0:
        raise SystemExit("%s failed with %d; see %s" % (label, rc, log))
    return took


def per_prompt_min_mass(target_dir, split):
    """prompt_id -> the smallest optimal probability of that prompt's solution."""
    npz = Path(target_dir) / ("%s_per_prompt.npz" % split)
    if npz.exists():
        z = np.load(npz, allow_pickle=True)
        return {str(p): float(pi.min()) for p, pi in zip(z["prompt_ids"], z["pi"])}
    meta = Path(target_dir) / split / "tensor/meta.json"
    star = Path(target_dir) / split / "solver/pi_star.npz"
    if meta.exists() and star.exists():
        ids = json.loads(meta.read_text())["prompt_ids"]
        pi = np.load(star)["pi"]
        return {str(p): float(pi[k].min()) for k, p in enumerate(ids)}
    raise SystemExit("no solution artifact under %s for %s" % (target_dir, split))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", required=True, help="us1 or ut1")
    ap.add_argument("--objectives", type=int, required=True)
    ap.add_argument("--split-tag", required=True, help="the frozen split dir, e.g. us_v1")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--config-dir", default=str(UF / "configs"))
    ap.add_argument("--queue-dir", default=str(UF / "jobs/queue"))
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--max-prompt-length", type=int, default=1024)
    ap.add_argument("--from-step", type=int, default=1)
    ap.add_argument("--to-step", type=int, default=7)
    args = ap.parse_args()

    p, K = args.panel, args.objectives
    certified = "%sp" % args.split_tag        # us_v1 -> us_v1p
    representable = "%sr" % args.split_tag    # us_v1 -> us_v1r
    log = SUB / ("%s_stage2.log" % p)
    record_path = SUB / ("%s_stage2.json" % p)
    record = json.loads(record_path.read_text()) if record_path.exists() else {"panel": p}

    def step(n):
        return args.from_step <= n <= args.to_step

    if step(1):
        run(["python3", str(CODE / "union_score_panel.py"), "--objectives", str(K),
             "--tag", "%s_psc" % p, "--judge-shards", str(args.shards),
             "--pool", p, "--pool-shards", str(args.shards),
             "--out", p, "--shards", str(args.shards)],
            SOLVE_ENV, "1_score", log)
        record["score_dir"] = str(UF / "scores" / p)

    if step(2):
        run(["python3", str(CODE / ("probe_feasible_%s.py" % p)),
             "--scores", str(UF / "scores" / p), "--shards", str(args.shards),
             "--splits", args.split_tag, "--out-splits", certified,
             "--beta", str(args.beta), "--eta", str(args.eta),
             "--workers", str(args.workers)],
            SOLVE_ENV, "2_probe", log)
        record["certified_split"] = certified

    # Table 3 asks for global NBPO and prompt-wise NBPO. The per-prompt max-min
    # solver is probed for feasibility (step 2 intersects all three rules) but is
    # not a Table 3 row, so it is not solved or trained here; PROSPER's row uses
    # PROSPER's own estimator in step 6, not this max-min surrogate.
    arms = [("pw_nbpo", "solve_pros4_pw_nbpo_%s.py" % p, "per_prompt"),
            ("nbpo", "solve_pros4_targets_%s.py" % p, "global")]

    failures = {}

    def solve(split_dir, suffix, skip_dataset):
        """Run every solver against one split directory."""
        out_names = {}
        for arm, script, kind in arms:
            name = "%s_%s%s" % (p, arm, suffix)
            out = UF / "targets" / name
            if out.exists():
                print(json.dumps({"skip": name, "why": "already solved"}), flush=True)
                out_names[arm] = str(out)
                continue
            cmd = ["python3", str(CODE / script), "--out-name", name,
                   "--shards", str(args.shards), "--beta", str(args.beta),
                   "--eta", str(args.eta), "--workers", str(args.workers)]
            if kind == "global":
                cmd += ["--splits", split_dir,
                        "--train-scores", str(UF / "scores" / p),
                        "--train-pool", str(UF / "pools" / p),
                        "--dev-scores", str(UF / "scores" / p),
                        "--dev-pool", str(UF / "pools" / p),
                        "--aggregation", "nash", "--representation", "adaptive_game"]
            if skip_dataset:
                cmd.append("--skip-dataset")
            env = dict(SOLVE_ENV if skip_dataset else WRITE_ENV)
            label = "%s_solve_%s" % ("3" if skip_dataset else "5", arm)
            try:
                run(cmd, env, label, log)
            except SystemExit as exc:
                # A solver that refuses to certify its solution is a measured
                # outcome, not a pipeline error: the global shared-weight rule is
                # known to leave held-out prompts uncertified on this pool, and
                # the instruction is to report that rather than relax a check to
                # get past it. The per-prompt Nash arm is the panel's spine, so
                # its failure still stops the run.
                if arm == "pw_nbpo":
                    raise
                failures[arm] = {"step": label, "detail": str(exc),
                                 "log": str(log),
                                 "consequence": ("this arm has no dataset, so its "
                                                 "table row stays unmeasured; it is "
                                                 "not replaced by a surrogate")}
                print(json.dumps({"arm_failed": arm, "detail": str(exc)}), flush=True)
                continue
            out_names[arm] = str(out)
        return out_names

    if step(3):
        # the per-prompt solvers read the certified split from their own source;
        # only the global solver takes it as a flag
        record["probe_solves"] = solve(certified, "_probe", True)

    if step(4):
        src = dest = UF / "splits" / certified
        detail, bad = {}, {"train": set(), "dev": set()}
        for arm, _, _ in arms:
            tdir = UF / "targets" / ("%s_%s_probe" % (p, arm))
            if arm in failures or not tdir.exists():
                detail[arm] = {"skipped": "this arm produced no probe solution"}
                continue
            detail[arm] = {}
            for split in ("train", "dev"):
                mass = per_prompt_min_mass(tdir, split)
                low = sorted(k for k, v in mass.items() if v < MIN_MASS)
                bad[split].update(low)
                detail[arm][split] = {"prompts": len(mass), "below_threshold": len(low),
                                      "min_mass_seen": min(mass.values()) if mass else None}
        written = {}
        for split, name in (("train", "policy_train"), ("dev", "policy_dev")):
            live = src / ("%s.jsonl" % name)
            keepsake = src / ("certified_before_representability_%s.jsonl" % name)
            if not keepsake.exists():          # preserve what the probe certified
                keepsake.write_bytes(live.read_bytes())
            rows = [json.loads(l) for l in keepsake.open() if l.strip()]
            kept = [r for r in rows if r["prompt_id"] not in bad[split]]
            with live.open("w") as f:
                for r in kept:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            written[name] = {"in": len(rows), "kept": len(kept),
                             "dropped": len(rows) - len(kept),
                             "certified_file_preserved_as": str(keepsake)}
        if not written["policy_train"]["kept"]:
            raise SystemExit("every train prompt is unrepresentable in some arm")
        record["representable"] = {
            "split": certified, "min_representable_mass": MIN_MASS,
            "unit_of_exclusion": "prompt",
            "rule": ("a prompt leaves EVERY arm if its optimal mass is below the "
                     "ten-decimal representable threshold in ANY arm, so the arms "
                     "stay on one prompt set"),
            "per_arm": detail, "splits": written}
        print(json.dumps(record["representable"]["splits"], indent=1), flush=True)

    if step(5):
        record["solves"] = solve(certified, "", False)

    if step(6):
        soft = UF / "datasets" / ("%s_soft_pavg" % p)
        if not soft.exists():
            run(["python3", str(CODE / "build_panel_softlabels.py"), "--panel", p,
                 "--source-dataset", str(UF / "datasets" / ("%s_pw_nbpo" % p)),
                 "--scores", str(UF / "scores" / p), "--shards", str(args.shards),
                 "--out", str(soft)], WRITE_ENV, "6_softlabels", log)
        pt = UF / "targets" / ("%s_prosper" % p)
        if not pt.exists():
            run(["python3", str(CODE / "prosper_targets_panel.py"),
                 "--objectives", str(K), "--tag", "%s_psc" % p,
                 "--shards", str(args.shards), "--splits", certified,
                 "--beta", str(args.beta), "--eta", str(args.eta),
                 "--out", "%s_prosper" % p], SOLVE_ENV, "6_prosper_targets", log)
        pd = UF / "datasets" / ("%s_prosper" % p)
        if not pd.exists():
            run(["python3", str(CODE / "build_prosper_dataset.py"), "--panel", p,
                 "--targets", "%s_prosper" % p, "--pool", p,
                 "--pool-shards", str(args.shards), "--out", str(pd)],
                WRITE_ENV, "6_prosper_dataset", log)
        record["datasets"] = {"soft_pavg": str(soft), "prosper": str(pd)}

    if step(7):
        record["queued"] = queue_training(p, args.epochs, record,
                                          args.max_length, args.max_prompt_length,
                                          Path(args.config_dir), Path(args.queue_dir))

    if failures:
        record["arm_failures"] = failures
    record_path.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"panel": p, "record": str(record_path),
                      "steps_run": [n for n in range(1, 8) if step(n)]}, indent=1))
    return 0


def dataset_rows(path, split):
    """Row count of one split, read through the trainer's own datasets version."""
    from datasets import load_from_disk
    return len(load_from_disk(path)[split])


ARM_SPECS = [
    # arm id, dataset suffix, loss, extra config. The three baseline rows keep
    # the hyperparameters their UW counterparts used, so a panel-to-panel
    # difference is the panel and not a retuning.
    ("nbpo", "nbpo", "nbpo", {}),
    ("pw_nbpo", "pw_nbpo", "nbpo", {}),
    ("dpo_soft", "soft_pavg", "dpo_soft", {"dpo_beta": 0.1}),
    ("inpo_soft", "soft_pavg", "inpo_soft", {"eta": 0.005, "ratio": 1.0 / 3.0,
                                             "inpo_prev_equals_reference": True}),
    ("prosper", "prosper", "ronpo", {"ronpo_alpha": 1.0, "ronpo_tau": 0.0,
                                     "ronpo_target_column": "ronpo_target",
                                     "eta": 1.0,
                                     "inpo_prev_equals_reference": True}),
]


def queue_training(panel, epochs, record, max_length, max_prompt_length,
                   config_dir, queue_dir):
    """One 4-GPU training spec per arm, at a step count matched across arms."""
    import yaml
    base = yaml.safe_load(Path(UF, "configs", "uw1_pw_nbpo.yaml").read_text())
    tokens_per_step = (base["per_device_train_batch_size"]
                       * base["gradient_accumulation_steps"] * 4)
    queued = {}
    for arm, suffix, loss, extra in ARM_SPECS:
        ds = UF / "datasets" / ("%s_%s" % (panel, suffix))
        if not ds.exists():
            print(json.dumps({"skip_arm": arm, "why": "no dataset at %s" % ds}), flush=True)
            continue
        rows = dataset_rows(str(ds), "train")
        steps = int(math.ceil(epochs * rows / tokens_per_step))
        cfg = dict(base)
        cfg["max_length"] = max_length
        cfg["max_prompt_length"] = max_prompt_length
        cfg.update(extra)
        cfg["loss_type"] = loss
        cfg["max_steps"] = steps
        cfg["eval_steps"] = steps
        cfg["save_steps"] = steps
        cfg["dataset_mixer"] = {str(ds): 1.0}
        cfg["output_dir"] = str(UF / "arms" / ("%s_%s" % (panel, arm)))
        cfg["run_name"] = "%s_%s" % (panel, arm)
        cfg["model_name_or_path"] = BASE_MODEL
        cfg["model_revision"] = BASE_REV
        cfg["nbpo_reference_model_path"] = BASE_MODEL
        # The reference-handling and immutable-token plumbing stays on for every
        # arm, exactly as the verified UW baseline configs have it: the soft-label
        # and PROSPER losses also read reference log-ratios and must score the
        # pool's own tokens. Only the target-column fields are NBPO-specific.
        for key in ("nbpo_expected_dataset_manifest_sha256",
                    "nbpo_expected_solver_artifact_sha256"):
            cfg.pop(key, None)
        if loss != "nbpo":
            for key in ("nbpo_target_mode", "nbpo_target_column", "nbpo_target_units",
                        "nbpo_eta_already_included"):
                cfg.pop(key, None)
        if loss == "dpo_soft":
            cfg.pop("eta", None)              # dpo_soft has no proximal step
        if loss == "nbpo":
            # pin the dataset manifest and the solver solution this arm was built
            # from, so a config cannot be pointed at a different solve later
            done = json.loads(Path(UF, "targets", "%s_%s" % (panel, arm),
                                   "complete.json").read_text())
            man = done.get("dataset_manifest_sha256")
            sol = (done["splits"]["train"].get("solver_solution_sha256")
                   if "splits" in done else None)
            if not man or not sol:
                raise SystemExit("%s: solve record has no manifest/solution hash" % arm)
            cfg["nbpo_expected_dataset_manifest_sha256"] = man
            cfg["nbpo_expected_solver_artifact_sha256"] = sol
        config_dir.mkdir(parents=True, exist_ok=True)
        path = config_dir / ("%s_%s.yaml" % (panel, arm))
        path.write_text(yaml.safe_dump(cfg, sort_keys=True))
        spec = {
            "job_id": "%s_train_%s" % (panel, arm), "priority": 120, "gpus": 4,
            "depends_on": [], "cwd": "%s/code" % DEPS,
            "env": {"PYTHONPATH": "%s/deps_train:%s/code" % (DEPS, DEPS),
                    "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "4", "MNPO_DISABLE_APEX": "1",
                    "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                    "WANDB_MODE": "disabled",
                    "VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
            "timeout_s": 86400,
            "command": ["python3", "-m", "torch.distributed.run", "--standalone",
                        "--nnodes=1", "--nproc_per_node=4", "-m",
                        "mnpo_scripts.run_mnpo", str(path)],
            "artifacts": [str(UF / "arms" / ("%s_%s" % (panel, arm)) / "config.json")],
            "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "note": ("%s panel, %s arm: %d train rows, %d steps for %d epochs at %d "
                     "sequences per step" % (panel.upper(), arm, rows, steps, epochs,
                                             tokens_per_step)),
        }
        queue_dir.mkdir(parents=True, exist_ok=True)
        (queue_dir / ("120_%s_train_%s.json" % (panel, arm))).write_text(
            json.dumps(spec, indent=1) + "\n")
        queued[arm] = {"rows": rows, "max_steps": steps, "config": str(path),
                       "loss_type": loss}
        print(json.dumps({arm: queued[arm]}), flush=True)
    return queued


if __name__ == "__main__":
    raise SystemExit(main())
