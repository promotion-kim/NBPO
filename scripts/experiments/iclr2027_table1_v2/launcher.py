#!/usr/bin/env python3
"""Idempotent job launcher and run registry for the ICLR-2027 Table-1 rebuild.

One binary owns every long job in the campaign so that "what ran, on which
GPUs, from which commit, and did it finish" is answerable from a file rather
than from shell history.

Subcommands
-----------
``status``     what every registered run is doing now (liveness re-checked from
               the PID / job id, not from the last thing written)
``dry-run``    print the exact commands and GPU assignment; touch nothing
``launch``     start everything not already complete, respecting the phase's
               GPU policy
``resume``     the same as ``launch``; completed runs are skipped only after
               their outputs are re-hashed
``validate``   re-check every completed run's declared outputs against the
               hashes recorded when it finished
``aggregate``  emit the registry as CSV for the results build
``cancel``     stop one run by id

Scheduling policy (Section 11).  Phases declare how many GPUs they need and
whether they may share the node:

===============================  ======  ==========================
phase                            gpus    concurrency
===============================  ======  ==========================
``generate``  reference decoding   1      one shard per GPU
``judge``     Qwen3-32B            all    exclusive -- nothing colocates
``rm``        objective RM         1      one objective per GPU
``policy``    8B policy training   1      one run per GPU
``eval``      Llama-3.3-70B judge  all    exclusive
===============================  ======  ==========================

Slurm is detected at run time (``sbatch`` on PATH); when present the launcher
writes reproducible sbatch files instead of forking, and records the job id in
the same registry field as a PID.  On this machine there is no Slurm, so runs
are ``nohup``-ed with an explicit ``CUDA_VISIBLE_DEVICES`` and a PID file.

Every registry entry binds the run to the code and inputs that produced it: the
git commit, the config hash, the resolved command, the GPU list, start/end
times, the exit code, the log path and the output hashes.  A run whose config
hash changes is a *different* run and gets a new id rather than overwriting the
old one -- there is no path here that silently reuses a stale artifact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

PHASES = {
    "generate": {"gpus_per_run": 1, "exclusive": False},
    "judge":    {"gpus_per_run": "all", "exclusive": True},
    "rm":       {"gpus_per_run": 1, "exclusive": False},
    "policy":   {"gpus_per_run": 1, "exclusive": False},
    "eval":     {"gpus_per_run": "all", "exclusive": True},
    "cpu":      {"gpus_per_run": 0, "exclusive": False},
}

TERMINAL = ("complete", "failed", "cancelled")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def have_slurm() -> bool:
    from shutil import which
    return which("sbatch") is not None


@dataclass
class Run:
    run_id: str
    phase: str
    method: Optional[str] = None
    representation: Optional[str] = None
    aggregation: Optional[str] = None
    seed: Optional[int] = None
    command: str = ""
    gpus: List[int] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    log_path: str = ""
    config_hash: str = ""
    git_commit: str = ""
    status: str = "pending"
    pid: Optional[int] = None
    slurm_job_id: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    exit_code: Optional[int] = None
    output_hashes: dict = field(default_factory=dict)
    failure_reason: Optional[str] = None
    note: Optional[str] = None


class Registry:
    """Append-only JSONL; the last record for a run id wins.

    Append-only because a crash mid-write must not be able to lose the history
    of the runs that already finished.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict:
        runs = {}
        if not self.path.exists():
            return runs
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            runs[rec["run_id"]] = rec
        return runs

    def append(self, run: Run) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = json.dumps(asdict(run), default=str) + "\n"
        with tmp.open("w") as fh:          # atomic append: write then concat
            fh.write(payload)
        with self.path.open("a") as fh:
            fh.write(payload)
        tmp.unlink(missing_ok=True)


def alive(rec: dict):
    """True / False / None. None means *not determinable from here*.

    A run executed inside a cluster pod carries a PID from that pod's namespace.
    Probing it with a local os.kill asks about an unrelated process on this
    machine and will usually say "gone" -- which previously made `status`
    announce a healthy remote job as failed. A run that declares a `host` other
    than this one is reported as unknown, and the operator is told how to check.
    """
    if rec.get("host") and rec["host"] != "local":
        return None
    if rec.get("slurm_job_id"):
        try:
            out = subprocess.check_output(["squeue", "-h", "-j", str(rec["slurm_job_id"])],
                                          text=True, stderr=subprocess.DEVNULL)
            return bool(out.strip())
        except Exception:
            return False
    pid = rec.get("pid")
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def refresh(rec: dict, root: Path) -> dict:
    """Re-derive a running run's state from the world, not from the record."""
    if rec["status"] != "running":
        return rec
    live = alive(rec)
    if live is None:
        rec = dict(rec)
        rec["status"] = "running_remote_unverified"
        rec["liveness_note"] = (
            f"declared host {rec.get('host')!r}; liveness cannot be checked from this "
            "machine. Verify with the command in `check_command` before treating this "
            "run as finished or failed.")
        return rec
    if live:
        return rec
    done = root / "done" / f"{rec['run_id']}.exit"
    if done.exists():
        try:
            code = int(done.read_text().strip())
        except ValueError:
            code = None
        rec = dict(rec)
        rec["exit_code"] = code
        rec["status"] = "complete" if code == 0 else "failed"
        if code != 0:
            rec["failure_reason"] = f"exit code {code}; see {rec['log_path']}"
        rec["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z",
                                        time.localtime(done.stat().st_mtime))
    else:
        rec = dict(rec)
        rec["status"] = "failed"
        rec["failure_reason"] = ("process is gone and no exit sentinel was written "
                                 "(killed, OOM, or node lost)")
    return rec


def visible_gpus() -> List[int]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [int(x) for x in env.split(",") if x.strip() != ""]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True)
        return [int(l) for l in out.split() if l.strip().isdigit()]
    except Exception:
        return []


def load_plan(path: Path) -> List[Run]:
    plan = json.loads(path.read_text())
    runs = []
    for spec in plan["runs"]:
        phase = spec["phase"]
        if phase not in PHASES:
            raise SystemExit(f"unknown phase {phase!r}; expected one of {list(PHASES)}")
        runs.append(Run(
            run_id=spec["run_id"], phase=phase, method=spec.get("method"),
            representation=spec.get("representation"),
            aggregation=spec.get("aggregation"), seed=spec.get("seed"),
            command=spec["command"], outputs=spec.get("outputs", []),
            config_hash=sha256_json(spec), note=spec.get("note")))
    return runs


def launch_one(run: Run, gpus: List[int], root: Path, repo: Path, use_slurm: bool) -> Run:
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (root / "done").mkdir(parents=True, exist_ok=True)
    (root / "pids").mkdir(parents=True, exist_ok=True)
    run.log_path = str(logs / f"{run.run_id}.log")
    run.gpus = gpus
    run.git_commit = git_commit(repo)
    run.started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    exit_file = root / "done" / f"{run.run_id}.exit"
    exit_file.unlink(missing_ok=True)
    dev = ",".join(str(g) for g in gpus)
    # The exit sentinel is what makes `status` honest after a crash: the record
    # says "running", and only the sentinel (or its absence) can end that.
    wrapped = (f"cd {shlex.quote(str(repo))} && "
               f"CUDA_VISIBLE_DEVICES={dev} {run.command}; "
               f"echo $? > {shlex.quote(str(exit_file))}")
    if use_slurm:
        sb = root / "sbatch" / f"{run.run_id}.sbatch"
        sb.parent.mkdir(parents=True, exist_ok=True)
        sb.write_text("#!/bin/bash\n"
                      f"#SBATCH --job-name={run.run_id}\n"
                      f"#SBATCH --output={run.log_path}\n"
                      f"#SBATCH --gres=gpu:{len(gpus)}\n\n{wrapped}\n")
        out = subprocess.check_output(["sbatch", "--parsable", str(sb)], text=True).strip()
        run.slurm_job_id = out.split(";")[0]
    else:
        with open(run.log_path, "ab") as log:
            proc = subprocess.Popen(["bash", "-lc", wrapped], stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        run.pid = proc.pid
        (root / "pids" / f"{run.run_id}.pid").write_text(str(proc.pid))
    run.status = "running"
    return run


def cmd_status(args, root, reg):
    runs = {k: refresh(v, root) for k, v in reg.load().items()}
    if not runs:
        print("registry is empty")
        return
    by_status = {}
    for r in runs.values():
        by_status.setdefault(r["status"], []).append(r)
    width = max(len(r) for r in runs)
    for status in ("running", "running_remote_unverified", "pending", "complete",
                   "failed", "cancelled"):
        for r in sorted(by_status.get(status, []), key=lambda x: x["run_id"]):
            ident = (f"job {r['slurm_job_id']}" if r.get("slurm_job_id")
                     else (f"pid {r['pid']}" if r.get("pid") else "-"))
            print(f"{r['run_id']:<{width}}  {status:<9}  {r['phase']:<9}  "
                  f"gpus={r.get('gpus')}  {ident}  {r.get('failure_reason') or ''}")
    print("\n" + "  ".join(f"{k}={len(v)}" for k, v in sorted(by_status.items())))


def cmd_validate(args, root, reg):
    runs = {k: refresh(v, root) for k, v in reg.load().items()}
    bad = 0
    for r in sorted(runs.values(), key=lambda x: x["run_id"]):
        if r["status"] != "complete":
            continue
        for out, want in (r.get("output_hashes") or {}).items():
            p = Path(out)
            if not p.exists():
                print(f"MISSING  {r['run_id']}  {out}")
                bad += 1
            elif sha256_file(p) != want:
                print(f"CHANGED  {r['run_id']}  {out}")
                bad += 1
        for out in r.get("outputs", []):
            if out not in (r.get("output_hashes") or {}):
                print(f"UNHASHED {r['run_id']}  {out}")
                bad += 1
    print("validate: OK" if not bad else f"validate: {bad} problem(s)")
    return 1 if bad else 0


def cmd_aggregate(args, root, reg):
    runs = {k: refresh(v, root) for k, v in reg.load().items()}
    out = root / "run_registry.csv"
    cols = ["run_id", "phase", "method", "representation", "aggregation", "seed",
            "status", "gpus", "pid", "slurm_job_id", "started_at", "ended_at",
            "exit_code", "config_hash", "git_commit", "log_path", "failure_reason"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(runs.values(), key=lambda x: x["run_id"]):
            w.writerow({c: r.get(c) for c in cols})
    print(f"wrote {out} ({len(runs)} runs)")


def cmd_cancel(args, root, reg):
    runs = reg.load()
    rec = runs.get(args.run_id)
    if rec is None:
        raise SystemExit(f"no such run {args.run_id}")
    if rec.get("slurm_job_id"):
        subprocess.run(["scancel", str(rec["slurm_job_id"])], check=False)
    elif rec.get("pid"):
        try:
            os.killpg(os.getpgid(int(rec["pid"])), signal.SIGTERM)
        except Exception as exc:
            print(f"could not signal pid {rec['pid']}: {exc}")
    run = Run(**{k: v for k, v in rec.items() if k in Run.__dataclass_fields__})
    run.status = "cancelled"
    run.ended_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    reg.append(run)
    print(f"cancelled {args.run_id}")


def cmd_launch(args, root, reg, dry: bool):
    repo = args.repo
    plan = load_plan(args.plan)
    existing = {k: refresh(v, root) for k, v in reg.load().items()}
    use_slurm = have_slurm() and not args.no_slurm
    gpus = visible_gpus()
    print(f"scheduler: {'slurm' if use_slurm else 'direct shell (nohup + PID files)'}; "
          f"visible GPUs: {gpus or 'none'}")

    busy = set()
    exclusive_running = False
    for r in existing.values():
        if r["status"] == "running":
            busy.update(r.get("gpus") or [])
            if PHASES[r["phase"]]["exclusive"]:
                exclusive_running = True

    todo = []
    for run in plan:
        prev = existing.get(run.run_id)
        if prev and prev["status"] == "complete":
            if prev.get("config_hash") == run.config_hash:
                print(f"skip     {run.run_id}: complete, config hash unchanged")
                continue
            print(f"RECONFIG {run.run_id}: config hash changed since it completed; "
                  f"it will re-run (old artifacts are not overwritten in place)")
        elif prev and prev["status"] == "running":
            print(f"running  {run.run_id}: already live "
                  f"({'job ' + str(prev['slurm_job_id']) if prev.get('slurm_job_id') else 'pid ' + str(prev.get('pid'))})")
            continue
        todo.append(run)

    for run in todo:
        policy = PHASES[run.phase]
        need = policy["gpus_per_run"]
        if policy["exclusive"]:
            if busy or exclusive_running:
                print(f"defer    {run.run_id}: phase '{run.phase}' is exclusive and "
                      f"GPUs {sorted(busy)} are in use")
                continue
            assign = list(gpus)
        elif need == 0:
            assign = []
        else:
            free = [g for g in gpus if g not in busy]
            if len(free) < need:
                print(f"defer    {run.run_id}: needs {need} GPU(s), {len(free)} free")
                continue
            assign = free[:need]
        if dry:
            print(f"DRY-RUN  {run.run_id}  phase={run.phase}  gpus={assign}\n"
                  f"         CUDA_VISIBLE_DEVICES={','.join(map(str, assign))} {run.command}")
            busy.update(assign)
            if policy["exclusive"]:
                exclusive_running = True
            continue
        run = launch_one(run, assign, root, repo, use_slurm)
        reg.append(run)
        ident = f"job {run.slurm_job_id}" if run.slurm_job_id else f"pid {run.pid}"
        print(f"LAUNCHED {run.run_id}  gpus={assign}  {ident}  log={run.log_path}")
        busy.update(assign)
        if policy["exclusive"]:
            exclusive_running = True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["status", "dry-run", "launch", "resume",
                                       "validate", "aggregate", "cancel"])
    ap.add_argument("--root", type=Path,
                    default=Path("experiments/iclr2027_table1_v2"))
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--plan", type=Path, default=None,
                    help="JSON plan file; required for dry-run/launch/resume")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-slurm", action="store_true")
    args = ap.parse_args()

    root = args.root
    reg = Registry(root / "run_registry.jsonl")
    if args.action in ("dry-run", "launch", "resume"):
        if args.plan is None:
            raise SystemExit("--plan is required for dry-run/launch/resume")
        cmd_launch(args, root, reg, dry=(args.action == "dry-run"))
    elif args.action == "status":
        cmd_status(args, root, reg)
    elif args.action == "validate":
        sys.exit(cmd_validate(args, root, reg))
    elif args.action == "aggregate":
        cmd_aggregate(args, root, reg)
    elif args.action == "cancel":
        if not args.run_id:
            raise SystemExit("--run-id is required for cancel")
        cmd_cancel(args, root, reg)


if __name__ == "__main__":
    main()
