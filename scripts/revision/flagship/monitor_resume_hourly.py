#!/usr/bin/env python3
"""Write measured, rank-free hourly snapshots for the four-B200 resume run."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def gpu_snapshot() -> list[dict]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    rows = []
    for line in output.splitlines():
        index, memory, utilization, temperature = [int(value.strip()) for value in line.split(",")]
        if index in range(4):
            rows.append({
                "gpu": index, "memory_used_mib": memory,
                "utilization_percent": utilization, "temperature_c": temperature,
            })
    return rows


def log_measurement(root: Path, method: str, seed: int) -> dict:
    logs = sorted((root / "logs").glob(f"full_{method}_s{seed}_a*.log"))
    if not logs:
        return {"progress_step": None, "error_matches": 0, "log": None}
    log = logs[-1]
    tail = log.read_bytes()[-500_000:].decode("utf-8", errors="replace").replace("\r", "\n")
    steps = [int(match) for match in re.findall(r"(?:^|\s)(\d{1,3})/900(?:\s|$)", tail)]
    error_pattern = re.compile(
        r"Traceback \(most recent call last\)|OutOfMemory|CUDA out of memory|nonfinite",
        re.IGNORECASE,
    )
    return {
        "progress_step": max(steps) if steps else None,
        "error_matches": len(error_pattern.findall(tail)),
        "log": str(log),
        "log_mtime": datetime.fromtimestamp(log.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
        "log_size": log.stat().st_size,
    }


def latest_job_status(root: Path, method: str, seed: int) -> dict:
    candidates = sorted((root / "full" / method / f"seed{seed}").glob("attempt*/job_status.json"))
    return read_json(candidates[-1]) if candidates else {}


def wandb_measurement(run_id: str | None) -> dict:
    if not run_id:
        return {"state": "missing_run_id", "latest_metrics": {}}
    try:
        import math
        import wandb

        run = wandb.Api(timeout=20).run(f"promotion-kim/mnpo/{run_id}")
        history = run.history(samples=20)
        wanted = [
            key for key in (
                "train/loss", "train/grad_norm", "train/learning_rate",
                "loss", "grad_norm", "learning_rate", "_step",
            ) if key in history.columns
        ]
        records = history[wanted].dropna(how="all").tail(1).to_dict("records")
        metrics = records[0] if records else {}
        finite = all(
            math.isfinite(float(value)) for value in metrics.values()
            if isinstance(value, (int, float))
        )
        return {"state": run.state, "latest_metrics": metrics, "finite": finite}
    except Exception as error:  # Monitoring must not terminate training on API failure.
        return {"state": "api_error", "latest_metrics": {}, "error": type(error).__name__}


def snapshot(root: Path) -> dict:
    dispatcher = read_json(root / "status/recovery_dispatch_4gpu.json")
    running = dispatcher.get("running", dispatcher.get("running_seed42_workers", []))
    jobs = []
    for row in running:
        method, seed = str(row["method"]), int(row["seed"])
        selected = read_json(root / "status/selected_workers" / f"{method}_s{seed}.json")
        job_status = latest_job_status(root, method, seed)
        jobs.append({
            **row, "selected_status": selected, "job_status": job_status,
            **log_measurement(root, method, seed),
            "wandb": wandb_measurement(job_status.get("wandb_run_id")),
        })
    ledger = root / "hf_uploads.jsonl"
    ledger_entries = sum(1 for line in ledger.read_text().splitlines() if line.strip()) if ledger.exists() else 0
    return {
        "measured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dispatcher_status": dispatcher.get("status", "missing"),
        "pending_count": len(dispatcher.get("pending", [])),
        "running": jobs,
        "finished": dispatcher.get("finished", []),
        "gpus": gpu_snapshot(),
        "hf_ledger_entries": ledger_entries,
        "hf_watcher": read_json(root / "status/hf_upload_watcher.json"),
        "ifeval": read_json(root / "eval/p2_ifeval_seed42/status.json"),
        "sealed_test_opened": False,
    }


def append_human_status(path: Path, value: dict) -> None:
    running = []
    errors = 0
    for row in value["running"]:
        step = row.get("progress_step")
        running.append(f"{row['method']}-s{row['seed']}={step if step is not None else 'loading'}/900")
        errors += int(row.get("error_matches", 0))
    gpu_text = ", ".join(
        f"GPU{row['gpu']} {row['memory_used_mib']}MiB/{row['utilization_percent']}%"
        for row in value["gpus"]
    )
    line = (
        f"- {value['measured_at']} — New-container hourly measured snapshot: "
        f"dispatcher={value['dispatcher_status']}, pending={value['pending_count']}, "
        f"running=[{', '.join(running)}], active-log error matches={errors}, "
        f"HF ledger={value['hf_ledger_entries']}; {gpu_text}. "
        "KTO is excluded by explicit scope amendment; sealed test remains unopened.\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=3600)
    parser.add_argument("--stop-at", required=True)
    args = parser.parse_args()
    stop_at = datetime.fromisoformat(args.stop_at).timestamp()
    latest = args.root / "status/hourly_monitor.json"
    history = args.root / "status/hourly_snapshots.jsonl"
    human = args.root / "overnight_status.md"
    while time.time() < stop_at:
        value = snapshot(args.root)
        atomic_json(latest, value)
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value) + "\n")
        append_human_status(human, value)
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
