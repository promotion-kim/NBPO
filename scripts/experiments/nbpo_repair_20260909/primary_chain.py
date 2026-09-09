"""Continue the fixed two-arm plan without depending on a live client stream.

This controller only waits for the already launched WBC job, validates its fixed
horizon/export, then runs the already frozen MSE config. No outcome-dependent
selection, job killing, retries, or horizon changes are permitted.
"""
import argparse
import datetime
import json
import os
import subprocess
import time
from pathlib import Path

from scripts.experiments.nbpo_repair_20260909.common import file_hash, read_jsonl, write_json


def completed_arm(root, arm):
    job = root / "jobs" / f"{arm}_primary_v1"
    exit_record = json.loads((job / "exit.json").read_text())
    if exit_record["exit_code"] != 0:
        raise ValueError(f"{arm} job failed; retain evidence and diagnose, do not auto-restart")
    output = root / "arms" / f"{arm}_primary_v1"
    state = json.loads((output / "trainer_state.json").read_text())
    if state["global_step"] != 1750:
        raise ValueError("A shorter checkpoint cannot be promoted to the fixed1750 comparison")
    for rank in range(4):
        rows = read_jsonl(output / f"runtime_rank{rank}.jsonl")
        if [r["step"] for r in rows] != list(range(1, 1751)):
            raise ValueError("Missing/duplicated actual optimizer updates")
        if any(r["master_dtypes"] != ["torch.float32"] or r["adam_state_dtypes"] != ["torch.float32"] for r in rows):
            raise ValueError("Runtime precision contract changed")
    weights = sorted(output.glob("*.safetensors"))
    if not weights or not (output / "config.json").exists():
        raise ValueError("Final full model was not exported")
    return {"arm": arm, "job_exit_sha256": file_hash(job / "exit.json"),
        "trainer_state_sha256": file_hash(output / "trainer_state.json"),
        "global_step": state["global_step"], "weight_sha256": {p.name: file_hash(p) for p in weights},
        "runtime_seconds": exit_record["seconds"], "gpu_hours_allocated": 4 * exit_record["seconds"] / 3600}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--background", action="store_true")
    args = ap.parse_args()
    directory = args.root / "controllers" / "primary_chain_v1"
    directory.mkdir(parents=True, exist_ok=True)
    if args.background:
        with (directory / "stdout.log").open("x") as log:
            command = ["python3", "-m", "scripts.experiments.nbpo_repair_20260909.primary_chain", "--root", str(args.root)]
            process = subprocess.Popen(command, cwd=args.root / "code", stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        write_json(directory / "launch.json", {"pid": process.pid, "command": command,
            "source_sha256": file_hash(__file__), "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        print(json.dumps({"controller_pid": process.pid, "wait_for": "wbc_primary_v1", "next": "mse_primary_v1"}), flush=True)
        return
    if (directory / "complete.json").exists() or (directory / "failure.json").exists():
        raise ValueError("Controller already reached a terminal state")
    try:
        while not (args.root / "jobs/wbc_primary_v1/exit.json").exists():
            time.sleep(15)
        wbc = completed_arm(args.root, "wbc")
        launch = json.loads((args.root / "jobs/wbc_primary_v1/launch.json").read_text())
        for name, expected in launch["source_hashes"].items():
            if name.startswith(("mnpo_scripts/", "alignment/")) or name == "scripts/simpo_trainer.py":
                if file_hash(args.root / "code" / name) != expected:
                    raise ValueError(f"Training source changed between matchedarms: {name}")
        frozen = json.loads((args.root / "protocols/primary_horizon_frozen_v1.json").read_text())
        config = Path(frozen["configs"]["mse"]["path"])
        if file_hash(config) != frozen["configs"]["mse"]["sha256"]:
            raise ValueError("Frozen MSE config changed")
        write_json(directory / "wbc_complete.json", wbc)
        command = ["python3", "-m", "scripts.experiments.nbpo_repair_20260909.train_job",
            "--root", str(args.root), "--config", str(config), "--job", "mse_primary_v1"]
        subprocess.run(command, check=True, cwd=args.root / "code")
        mse = completed_arm(args.root, "mse")
        write_json(directory / "complete.json", {"arms": [wbc, mse],
            "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "remaining": "Saved-response generation, evaluation and final manuscript/report; not overall campaign completion"})
    except Exception as exc:
        write_json(directory / "failure.json", {"error_type": type(exc).__name__, "message": str(exc),
            "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    main()
