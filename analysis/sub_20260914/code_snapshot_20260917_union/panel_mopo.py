"""Build and queue the MOPO row of Table 3 for one union panel.

MOPO's policy step is already in the trainer (`loss_type == "mopo"` is
importance-weighted behaviour cloning, `-rho(y) log pi(y)`), and this campaign
already has the three scripts that produce rho: the Eq. (3) dual solve, the
one-row-per-candidate builder, and the dataset materializer. All three are
reused as they are; the only generated piece is a panel copy of the dual solve,
whose objective names come from the panel's own base module instead of UF-4's.

The primary objective is fixed positionally, as the paper fixes it -- it writes
p_K for "the K-th (the primary) objective" and takes r_1 as primary in every
experiment -- so it is item0 of the panel's declared order: helpfulness on US,
summary preference on UT. It is NOT chosen by which choice would flatter the
baseline, and the constrained objective is whatever remains.

One ordering wart is handled explicitly rather than worked around. The row
builder hashes `train/solver/solution.json` to pin the solver artifact, but
MOPO's solve writes `rho.npz` and it is the materializer that later synthesizes
that file. Writing it here, immediately after the solve and with exactly the
content the materializer would write -- a record that binds to the rho file it
came from -- keeps the builder's pin meaningful instead of leaving it to run
against a file that does not exist yet.
"""
from __future__ import annotations

import argparse, hashlib, importlib, json, os, py_compile, subprocess, sys, time
from pathlib import Path

UF = Path("/work/uf4_20260910")
SUB = Path("/work/sub_20260914")
CODE = SUB / "code"
UFCODE = UF / "code"
DEPS = "/work/nbpo_repair_20260909"
BASE_MODEL = "/work/models/bases/Qwen2.5-7B-Instruct"
BASE_REV = "a09a35458c702b33eeacc393d103063234e8bc28"

SOLVE_ENV = {"PYTHONPATH": "%s/code:%s:%s" % (DEPS, UFCODE, CODE),
             "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
             "HF_HUB_OFFLINE": "1"}
WRITE_ENV = {"PYTHONPATH": "%s/deps_train:%s/code:%s:%s" % (DEPS, DEPS, UFCODE, CODE),
             "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
             "HF_HUB_OFFLINE": "1"}


def fh(p):
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def run(cmd, env_extra, label, log, cwd):
    env = dict(os.environ)
    env.update({k: str(v) for k, v in env_extra.items()})
    t0 = time.monotonic()
    with open(log, "a") as sink:
        sink.write("\n===== %s =====\n%s\n" % (label, " ".join(cmd)))
        sink.flush()
        rc = subprocess.call(cmd, env=env, cwd=str(cwd), stdout=sink,
                             stderr=subprocess.STDOUT)
    print(json.dumps({"step": label, "returncode": rc,
                      "seconds": round(time.monotonic() - t0, 1)}), flush=True)
    if rc != 0:
        raise SystemExit("%s failed with %d; see %s" % (label, rc, log))


def generate_solver(panel):
    """Panel copy of the Eq. (3) dual solve: objective names, nothing else."""
    src = UFCODE / "solve_mopo_targets.py"
    t = src.read_text()
    old = "import solve_uf4_targets as base"
    if t.count(old) != 1:
        raise SystemExit("base import anchor occurs %d times" % t.count(old))
    t = t.replace(old, "import solve_pros4_targets_%s as base" % panel)
    head = ("# Generated from solve_mopo_targets.py by panel_mopo.py -- do not edit by\n"
            "# hand. source sha256 %s\n"
            "# change: base module -> solve_pros4_targets_%s, so OBJECTIVES are this\n"
            "#         panel's declared objectives. tau, beta, the epsilon treatment and\n"
            "#         the dual solve are untouched.\n" % (fh(src), panel))
    dst = CODE / ("solve_mopo_targets_%s.py" % panel)
    dst.write_text(head + t)
    py_compile.compile(str(dst), doraise=True)
    sys.path.insert(0, str(CODE))
    sys.path.insert(0, str(UFCODE))
    importlib.invalidate_caches()
    mod = importlib.import_module(dst.stem)
    return dst, list(mod.OBJECTIVES)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--tensor-from", required=True,
                    help="the global solve whose tensors and pairs MOPO reads")
    ap.add_argument("--epsilon-mode", default="zero",
                    choices=("zero", "paper", "calibrated"))
    ap.add_argument("--tau", type=float, default=0.08)
    ap.add_argument("--beta", type=float, default=0.9995)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--priority", type=int, default=121)
    args = ap.parse_args()

    p = args.panel
    log = SUB / ("%s_mopo.log" % p)
    solver, objectives = generate_solver(p)
    primary = objectives[0]
    out_name = "%s_mopo" % p
    target_dir = UF / "targets" / out_name

    if not (target_dir / "complete.json").exists():
        run(["python3", str(solver), "--tensor-from", str(UF / "targets" / args.tensor_from),
             "--out-name", out_name, "--primary", primary,
             "--tau", str(args.tau), "--beta", str(args.beta),
             "--epsilon-mode", args.epsilon_mode],
            SOLVE_ENV, "1_mopo_dual", log, UFCODE)

    # the row builder pins train/solver/solution.json; MOPO's solve writes
    # rho.npz, so bind one to the other here instead of leaving the pin dangling
    solver_dir = target_dir / "train/solver"
    solver_dir.mkdir(parents=True, exist_ok=True)
    solution = solver_dir / "solution.json"
    if not solution.exists():
        rho_path = target_dir / "train/rho.npz"
        complete = json.loads((target_dir / "complete.json").read_text())
        solution.write_text(json.dumps({
            "aggregation": "mopo_constrained",
            "note": ("MOPO has no global weight vector: the dual is lambda over the "
                     "secondary objectives and the trained quantity is the per-candidate "
                     "importance ratio rho(y). This file exists so the shared job "
                     "generator can hash the solver artifact, and it binds to the rho "
                     "file it was produced from."),
            "rho_path": str(rho_path), "rho_sha256": fh(rho_path),
            "primary_objective": primary,
            "declarations": complete.get("declarations")}, indent=2) + "\n")

    if not (target_dir / "pairs/train.jsonl").exists():
        run(["python3", str(UFCODE / "build_mopo_rows.py"),
             "--rho-from", out_name, "--pairs-from", args.tensor_from,
             "--out-name", out_name],
            SOLVE_ENV, "2_mopo_rows", log, UFCODE)

    dataset = UF / "datasets" / out_name
    if not dataset.exists():
        run(["python3", str(UFCODE / "materialize_mopo_dataset.py"),
             "--targets", out_name, "--out-name", out_name,
             "--base-model", BASE_MODEL, "--model-revision", BASE_REV],
            WRITE_ENV, "3_mopo_dataset", log, UFCODE)

    # training spec, at the same recipe every other arm of this panel uses
    import yaml
    from datasets import load_from_disk
    base_cfg = yaml.safe_load((UF / "configs" / ("%s_pw_nbpo.yaml" % p)).read_text())
    rows = len(load_from_disk(str(dataset))["train"])
    per_step = base_cfg["per_device_train_batch_size"] * base_cfg["gradient_accumulation_steps"] * 4
    import math
    steps = int(math.ceil(args.epochs * rows / per_step))
    cfg = dict(base_cfg)
    cfg.update({"loss_type": "mopo", "max_steps": steps, "eval_steps": steps,
                "save_steps": steps,
                "dataset_mixer": {str(dataset): 1.0},
                "output_dir": str(UF / "arms" / ("%s_mopo" % p)),
                "run_name": "%s_mopo" % p,
                "ronpo_target_column": "ronpo_target",
                "nbpo_target_mode": "mopo_rho"})
    for key in ("nbpo_target_column", "nbpo_target_units", "nbpo_eta_already_included"):
        cfg.pop(key, None)
    # The trainer refuses a canonical config that does not pin its dataset
    # manifest, and it is right to: an unpinned config can be pointed at a
    # different materialization later. Both hashes already exist -- the
    # materializer records the manifest in the target's complete.json, and
    # solution.json binds to the rho file this arm was built from -- so they are
    # pinned here rather than dropped.
    done = json.loads((target_dir / "complete.json").read_text())
    manifest = done.get("dataset_manifest_sha256")
    if not manifest:
        raise SystemExit("%s has no dataset_manifest_sha256; run the materializer "
                         "before queueing" % (target_dir / "complete.json"))
    cfg["nbpo_expected_dataset_manifest_sha256"] = manifest
    cfg["nbpo_expected_solver_artifact_sha256"] = fh(solution)
    cfg_path = UF / "configs" / ("%s_mopo.yaml" % p)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True))
    spec = {
        "job_id": "%s_train_mopo" % p, "priority": args.priority, "gpus": 4,
        "depends_on": [], "cwd": "%s/code" % DEPS,
        "env": {"PYTHONPATH": "%s/deps_train:%s/code" % (DEPS, DEPS),
                "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "4", "MNPO_DISABLE_APEX": "1",
                "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                "WANDB_MODE": "disabled", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
        "timeout_s": 86400,
        "command": ["python3", "-m", "torch.distributed.run", "--standalone",
                    "--nnodes=1", "--nproc_per_node=4", "-m",
                    "mnpo_scripts.run_mnpo", str(cfg_path)],
        "artifacts": [str(UF / "arms" / ("%s_mopo" % p) / "config.json")],
        "config_sha256": fh(cfg_path),
        "note": ("%s panel, MOPO row: one row per (prompt, candidate) carrying the "
                 "Eq. (3) importance ratio rho(y); primary objective %s fixed "
                 "positionally; %d train rows, %d steps for %d epochs"
                 % (p.upper(), primary, rows, steps, args.epochs)),
    }
    (UF / "jobs/queue" / ("%d_%s_train_mopo.json" % (args.priority, p))).write_text(
        json.dumps(spec, indent=1) + "\n")
    print(json.dumps({"panel": p, "primary_objective": primary,
                      "objectives": objectives, "rows": rows, "max_steps": steps,
                      "dataset": str(dataset), "config": str(cfg_path),
                      "queued": spec["job_id"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
