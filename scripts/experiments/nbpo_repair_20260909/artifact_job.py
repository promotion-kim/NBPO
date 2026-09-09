"""Run one explicitly GPU-scoped evaluation job with persistent provenance.

This is a job wrapper, not a GPU scheduler. The caller must first reserve the
listed GPUs. It never retries, kills another job, downloads models, or changes
the frozen training plan. Only campaign evaluation modules are accepted.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import time
import zipfile
from pathlib import Path

from scripts.experiments.nbpo_repair_20260909.common import digest, file_hash, write_json

PREFIX = "scripts.experiments.nbpo_repair_20260909."
MODULES = {PREFIX + name for name in (
    "generate_eval", "evaluate_responses", "evaluate_saferlhf", "score_teacher_rm",
    "score_xstest_refusal", "validate_export")}


def validated_command(command):
    command = command[1:] if command and command[0] == "--" else command
    if len(command) < 3 or command[:2] != ["python3", "-m"] or command[2] not in MODULES:
        raise ValueError("Only explicitly listed task evaluation modules may be launched")
    return command


def runtime_environment(root, stack, gpu):
    code = f"{root}/code"
    stacks = {
        "train": f"{root}/deps_train:{code}",
        "eval": f"{code}:/work/pylibs_eval:/work/pylibs_ifeval:/work/nbpo_downstream_local_v1/code",
        "harmbench": f"{root}/deps_harmbench/site:{code}:/work/pylibs_eval:/work/pylibs_ifeval:/work/nbpo_downstream_local_v1/code",
    }
    if stack not in stacks or gpu not in ("cpu", "0", "1", "2", "3"):
        raise ValueError("A known dependency stack and at most one authorized GPU are required")
    return {"PYTHONPATH": stacks[stack], "CUDA_VISIBLE_DEVICES": "" if gpu == "cpu" else gpu,
        "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "4",
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "WANDB_MODE": "disabled", "MNPO_DISABLE_APEX": "1",
        "VLLM_CACHE_ROOT": f"{root}/cache/evaluation_gpu{gpu}"}


def wait_gpu_idle(gpu, *, timeout=60., poll=1., query=None, clock=time.monotonic, sleep=time.sleep):
    """Drain only the assigned GPU; never kill lingering engine descendants."""
    if gpu == "cpu":
        return {"gpu": gpu, "wait_seconds": 0., "checks": 0}
    if query is None:
        def query(target):
            return subprocess.check_output(["nvidia-smi", "--id=" + target,
                "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    started, checks = clock(), 0
    while True:
        output = query(gpu).strip()
        checks += 1
        if not output:
            return {"gpu": gpu, "wait_seconds": clock()-started, "checks": checks}
        if any(not line.strip().isdigit() for line in output.splitlines()):
            raise ValueError("Unrecognized GPU process query output; cannot prove idle")
        if clock() - started >= timeout:
            raise TimeoutError(f"GPU {gpu} still has compute PID(s) {output!r}; inspect, never kill")
        sleep(poll)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--gpu", choices=("cpu", "0", "1", "2", "3"), required=True)
    ap.add_argument("--stack", choices=("train", "eval", "harmbench"), required=True)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if args.root != Path("/work/nbpo_repair_20260909") or Path(args.job).name != args.job:
        raise ValueError("Unexpected campaign root or non-simple job name")
    command = validated_command(args.command)
    env_add = runtime_environment(args.root, args.stack, args.gpu)
    out = args.root / "jobs" / args.job
    out.mkdir(parents=True, exist_ok=False)
    code = args.root / "code"
    sources = {}
    with zipfile.ZipFile(out / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for folder in ("mnpo_scripts", "scripts/experiments/nbpo_repair_20260909",
                       "scripts/experiments/nbpo_downstream_local_v1"):
            for path in sorted((code / folder).rglob("*.py")):
                name, payload = str(path.relative_to(code)), path.read_bytes()
                sources[name] = digest(payload)
                archive.writestr(name, payload)
    env = dict(os.environ)
    env.update(env_add)
    version_command = ["python3", "-c", "import importlib.metadata as m, importlib.util as u, json; "
        "names=['torch','transformers','safetensors','numpy','vllm']; "
        "print(json.dumps({name:{'version':m.version(name),'module_path':u.find_spec(name).origin} for name in names}))"]
    versions = json.loads(subprocess.check_output(version_command, env=env, cwd=code, text=True))
    write_json(out / "environment_versions.json", versions)
    spec = {"command": command, "environment_overrides": env_add, "cwd": str(code),
        "gpu_count": int(args.gpu != "cpu"), "gpu_id": args.gpu,
        "source_hashes": sources, "source_archive_sha256": file_hash(out / "source.zip"),
        "environment_versions_sha256": file_hash(out / "environment_versions.json"),
        "paid_judge_api_calls": 0, "wrapper_source_sha256": file_hash(__file__)}
    write_json(out / "spec.json", spec)
    started = time.monotonic()
    before = wait_gpu_idle(args.gpu)
    with (out / "stdout.log").open("x") as log:
        process = subprocess.Popen(command, cwd=code, env=env, stdout=log,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        write_json(out / "launch.json", {**spec, "child_pid": process.pid,
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        child_rc = process.wait()
    rc, after, drain_error = child_rc, None, None
    try:
        after = wait_gpu_idle(args.gpu)
    except Exception as exc:
        drain_error = {"error_type": type(exc).__name__, "message": str(exc)}
        rc = child_rc or 97
    seconds = time.monotonic() - started
    write_json(out / "exit.json", {"exit_code": rc, "child_exit_code": child_rc, "seconds": seconds,
        "gpu_idle_before": before, "gpu_idle_after": after, "gpu_drain_error": drain_error,
        "gpu_count": spec["gpu_count"], "allocated_gpu_hours": spec["gpu_count"] * seconds / 3600,
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "log_sha256": file_hash(out / "stdout.log"), "paid_judge_api_calls": 0})
    print(f"{args.job}: exit={rc}, seconds={seconds:.2f}", flush=True)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
