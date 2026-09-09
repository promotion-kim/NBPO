"""One scoped four-H200 torchrun job with persistent launch/exit provenance."""
import argparse
import datetime
import json
import os
import subprocess
import time
import zipfile
from pathlib import Path

from scripts.experiments.nbpo_repair_20260909.common import digest, file_hash, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--job", required=True)
    args = ap.parse_args()
    if Path(args.job).name != args.job:
        raise ValueError("Job must be a simple directory name")
    out = args.root / "jobs" / args.job
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    code = args.root / "code"
    env_add = {"PYTHONPATH": f"{args.root}/deps_train:{code}",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "4", "MNPO_DISABLE_APEX": "1", "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "WANDB_MODE": "disabled"}
    env = os.environ.copy()
    env.update(env_add)
    command = ["python3", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        "--nproc_per_node=4", "-m", "mnpo_scripts.run_mnpo", str(args.config)]
    sources = {}
    with zipfile.ZipFile(out / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for folder in ("mnpo_scripts", "scripts", "alignment", "training_configs/nbpo/repair_20260909"):
            for path in sorted((code / folder).rglob("*")):
                if path.is_file() and path.suffix in (".py", ".yaml", ".json"):
                    name = str(path.relative_to(code))
                    payload = path.read_bytes()
                    sources[name] = digest(payload)
                    archive.writestr(name, payload)
        archive.writestr("resolved_run_config.yaml", args.config.read_bytes())
    launch = {"command": command, "cwd": str(code), "environment_overrides": env_add,
        "config": str(args.config), "config_sha256": file_hash(args.config),
        "source_hashes": sources, "source_archive_sha256": file_hash(out / "source.zip"),
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    started = time.monotonic()
    with (out / "stdout.log").open("x") as log:
        process = subprocess.Popen(command, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT)
        launch["child_pid"] = process.pid
        write_json(out / "launch.json", launch)
        result = process.wait()
    write_json(out / "exit.json", {"exit_code": result, "seconds": time.monotonic() - started,
        "gpu_count": 4, "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "log_sha256": file_hash(out / "stdout.log")})
    print(json.dumps({"job": args.job, "exit_code": result}), flush=True)
    raise SystemExit(result)


if __name__ == "__main__":
    main()
