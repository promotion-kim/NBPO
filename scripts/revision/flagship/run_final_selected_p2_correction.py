#!/usr/bin/env python3
"""Re-evaluate only RONPO P2 cells if P3 selects a different exact revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def wait_json(path: Path, terminal: set[str], stop_at: float) -> dict:
    while time.time() < stop_at:
        if path.exists():
            payload = json.loads(path.read_text())
            if payload.get("status") in terminal:
                return payload
        time.sleep(20)
    raise TimeoutError(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--stop-at", required=True)
    args = parser.parse_args()
    stop_at = datetime.fromisoformat(args.stop_at).timestamp()
    work = args.root / "eval/p2_final_selected_correction"
    status_path = work / "status.json"
    work.mkdir(parents=True, exist_ok=True)

    selection_path = args.root / "eval/p1_validation_reward_seed42/final_model_selection.json"
    selection = wait_json(selection_path, {"selected"}, stop_at)
    current_manifest = json.loads(
        (args.root / "eval/p1_validation_reward_seed42/model_manifest.json").read_text()
    )
    current = current_manifest["models"]["ronpo_full_expect"]
    selected = selection["selected_model"]
    if selected["revision"] == current["revision"] and selected["repo_id"] == current["repo_id"]:
        atomic_json(args.root / "eval/p2_final_model_override.json", {
            "status": "completed", "changed": False, "ronpo_full_expect": selected,
            "reason": "Final selected revision equals the already-evaluated revision.",
        })
        atomic_json(status_path, {"status": "completed", "changed": False})
        return

    atomic_json(status_path, {"status": "waiting", "stage": "sealed_p1_completion",
                              "changed": True, "selected": selected})
    sealed = wait_json(
        args.root / "eval/p1_sealed_reward_seed42/status.json", {"completed", "failed"}, stop_at
    )
    if sealed.get("status") != "completed":
        atomic_json(status_path, {"status": "blocked", "reason": "sealed P1 failed", "sealed": sealed})
        return

    ifeval = args.root / "eval/p2_ifeval_seed42"
    academic = args.root / "eval/p2_academic_seed42"
    revision_tag = current["revision"][:12]
    old_ifeval = ifeval / "raw/ronpo_full_expect"
    old_academic = academic / "raw/ronpo_full_expect"
    if old_ifeval.exists():
        destination = ifeval / "superseded" / f"ronpo_full_expect_{revision_tag}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.move(str(old_ifeval), str(destination))
    if old_academic.exists():
        destination = academic / "superseded" / f"ronpo_full_expect_{revision_tag}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.move(str(old_academic), str(destination))

    output = ifeval / "raw/ronpo_full_expect"
    output.mkdir(parents=True, exist_ok=True)
    identifier = "r" + hashlib.sha256(
        f"aaai27-p2-final-ifeval|{selected['revision']}".encode()
    ).hexdigest()[:11]
    model_args = ",".join((
        f"pretrained={selected['repo_id']}", f"revision={selected['revision']}",
        "dtype=bfloat16", "gpu_memory_utilization=0.88", "max_model_len=32768",
        "enable_thinking=False",
    ))
    command = [
        args.python, "-m", "lm_eval", "run", "--model", "vllm",
        "--model_args", model_args, "--tasks", "ifeval", "--num_fewshot", "0",
        "--batch_size", "auto", "--max_batch_size", "256", "--apply_chat_template",
        "--seed", "42,42,42,42", "--cache_requests", "true", "--show_config",
        "--output_path", str(output),
        "--wandb_args", "entity=promotion-kim", "project=mnpo", f"id={identifier}",
        "name=aaai27-p2-ifeval-final-ronpo-s42", "group=flagship_p2_seed42",
        "job_type=lm_eval", "resume=allow",
        "--wandb_config_args", "flagship_stage=P2_final_selected_IFEval",
        "model_name=ronpo_full_expect", "task_group=ifeval", "num_fewshot=0",
        "enable_thinking=False",
    ]
    env = os.environ.copy(); env.update({
        "CUDA_VISIBLE_DEVICES": "0", "HF_HOME": str(args.root / "cache/huggingface"),
        "HF_HUB_CACHE": str(args.root / "cache/huggingface/hub"),
        "HF_DATASETS_CACHE": str(args.root / "cache/huggingface/datasets"),
        "WANDB_MODE": "online", "WANDB_ENTITY": "promotion-kim", "WANDB_PROJECT": "mnpo",
        "TOKENIZERS_PARALLELISM": "false", "TORCH_CUDNN_SDPA_ENABLED": "0",
        "VLLM_HOST_IP": "127.0.0.1", "VLLM_PORT": "65100",
        "VLLM_DP_MASTER_PORT": "65101", "MASTER_PORT": "65102",
    })
    atomic_json(status_path, {"status": "running", "stage": "final_selected_ifeval",
                              "wandb_run_id": identifier, "selected": selected})
    with (work / "ifeval.log").open("a") as handle:
        subprocess.run(command, cwd=args.project, env=env, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)

    result_jsons = list(output.rglob("results_*.json"))
    if len(result_jsons) != 1:
        raise RuntimeError(f"expected one selected IFEval result, got {len(result_jsons)}")
    target = academic / "raw/ronpo_full_expect/ifeval_reused" / result_jsons[0].name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result_jsons[0], target)

    provenance_path = ifeval / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    replacement = {"method": "ronpo_full_expect", "seed": 42, **selected,
                   "selection": "final non-sealed validation selection"}
    provenance["models"] = [
        replacement if row.get("method") == "ronpo_full_expect" else row
        for row in provenance.get("models", [])
    ]
    provenance["final_selected_override"] = replacement
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    subprocess.run([
        args.python, str(args.project / "scripts/revision/flagship/aggregate_seed42_ifeval.py"),
        "--work", str(ifeval), "--output-dir", str(ifeval / "results"),
    ], cwd=args.project, check=True)
    override = {
        "status": "completed", "changed": True, "ronpo_full_expect": selected,
        "superseded_revision": current, "ifeval_wandb_run_id": identifier,
        "ifeval_source_json": str(result_jsons[0]),
    }
    atomic_json(args.root / "eval/p2_final_model_override.json", override)
    atomic_json(status_path, override)


if __name__ == "__main__":
    main()
