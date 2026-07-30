#!/usr/bin/env python3
"""Continuously upload each stability-passing P1 checkpoint to a public HF repo."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def completed_models(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    result = []
    for status_path in root.glob("full/*/seed*/attempt*/job_status.json"):
        status = json.loads(status_path.read_text())
        model = status_path.parent
        gate_path = model / "stability/gate.json"
        if status.get("status") != "completed" or not gate_path.exists():
            continue
        if json.loads(gate_path.read_text()).get("passed") is not True:
            continue
        result.append((model, status))
    return sorted(result, key=lambda item: (item[1]["method"], int(item[1]["seed"])))


def prepare_metadata(root: Path, project: Path, model: Path, status: dict[str, Any]) -> None:
    copies = {
        root / "data/split_manifest.json": model / "split_manifest.json",
        root / "pairs/pair_manifest.json": model / "pair_manifest.json",
        project / "results/ronpo_flagship_20260712/objective_protocol.json": model / "objective_protocol.json",
        project / "results/ronpo_flagship_20260712/resource_profile.json": model / "resource_profile.json",
        root / "scope_amendment_20260713.json": model / "scope_amendment_20260713.json",
        model / "stability/gate.json": model / "stability_gate.json",
        model / "job_status.json": model / "job_status.json",
    }
    for source, destination in copies.items():
        if source.resolve() == destination.resolve():
            continue
        if source.exists():
            shutil.copy2(source, destination)
    provenance = {
        "method": status["method"], "seed": status["seed"], "attempt": status["attempt"],
        "optimizer_steps": status["optimizer_steps"], "effective_batch_size": status["effective_batch_size"],
        "wandb_run_id": status["wandb_run_id"], "wandb_url": status["wandb_url"],
        "split_manifest": "split_manifest.json", "objective_protocol": "objective_protocol.json",
        "resource_profile": "resource_profile.json",
        "stability_gate": "stability_gate.json",
        "scope_amendment": (
            "scope_amendment_20260713.json"
            if (root / "scope_amendment_20260713.json").exists() else None
        ),
    }
    (model / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (model / "run_status.json").write_text(json.dumps(provenance | {"status": "completed"}, indent=2) + "\n")


def upload(args: argparse.Namespace, model: Path, status: dict[str, Any]) -> bool:
    upload_status = model / "hf_upload_status.json"
    if upload_status.exists() and json.loads(upload_status.read_text()).get("status") == "completed":
        current = json.loads(upload_status.read_text())
        # `attempt` denotes the training attempt. Keep the independent upload
        # retry counter explicit so a stabilized attempt-2/3 model is never
        # mislabeled as training attempt 1.
        if current.get("training_attempt") != status["attempt"]:
            atomic_json(upload_status, {
                "status": "completed", "repo_id": current["repo_id"],
                "attempt": status["attempt"], "training_attempt": status["attempt"],
                "upload_attempt": current.get("upload_attempt", current.get("attempt", 1)),
            })
        return True
    prepare_metadata(args.root, args.project, model, status)
    repo = f"promotion/qwen3-8b-aaai27-flagship-{status['method'].replace('_', '-')}-s{status['seed']}"
    command = [
        args.python, str(args.project / "scripts/revision/upload_checkpoint_to_hf.py"),
        "--local-path", str(model), "--repo-id", repo, "--method", status["method"],
        "--base-model", "Qwen/Qwen3-8B", "--seed", str(status["seed"]),
        "--notes", "AAAI-27 P1 matched budget: 900 optimizer steps, effective batch 16; passed non-thinking and collapse stability gates; sealed-test metrics are not used for checkpoint selection.",
        "--ledger", str(args.root / "hf_uploads.jsonl"),
    ]
    log = model / "hf_upload.log"
    for upload_attempt in range(1, 4):
        status_fields = {
            "repo_id": repo, "attempt": status["attempt"],
            "training_attempt": status["attempt"], "upload_attempt": upload_attempt,
        }
        atomic_json(upload_status, {"status": "running", **status_fields})
        with log.open("a", encoding="utf-8") as handle:
            rc = subprocess.run(command, cwd=args.project, stdout=handle, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            atomic_json(upload_status, {"status": "completed", **status_fields})
            # The root final model is sufficient after remote integrity checks;
            # remove only redundant Trainer checkpoints within this sjkim run.
            for checkpoint in model.glob("checkpoint-*"):
                if checkpoint.is_dir():
                    shutil.rmtree(checkpoint)
            return True
        atomic_json(upload_status, {"status": "failed", **status_fields, "returncode": rc})
        time.sleep(15)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    watcher_status = args.root / "status/hf_upload_watcher.json"
    args.root.joinpath("status").mkdir(parents=True, exist_ok=True)
    failures: dict[str, str] = {}
    while True:
        models = completed_models(args.root)
        for model, status in models:
            key = f"{status['method']}__s{status['seed']}"
            if key in failures:
                continue
            if not upload(args, model, status):
                failures[key] = str(model)
        s3 = args.root / "status/s3.json"
        final = s3.exists() and str(json.loads(s3.read_text()).get("status", "")).startswith("completed")
        pending = []
        for model, status in completed_models(args.root):
            upload_status = model / "hf_upload_status.json"
            if not upload_status.exists() or json.loads(upload_status.read_text()).get("status") != "completed":
                pending.append(f"{status['method']}__s{status['seed']}")
        atomic_json(watcher_status, {
            "status": "failed" if final and failures else "completed" if final and not pending else "running",
            "uploaded": len(models) - len(pending), "pending": pending, "failures": failures,
            "training_final": final, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        if final and (not pending or failures):
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
