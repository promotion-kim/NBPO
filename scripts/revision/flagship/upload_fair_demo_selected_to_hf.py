#!/usr/bin/env python3
"""Upload validation-selected fair-demo checkpoints to verified public HF repositories."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoConfig


def model_present(path: Path) -> bool:
    return (path / "model.safetensors").is_file() or (path / "model.safetensors.index.json").is_file()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--fair-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--namespace", default="promotion")
    args = parser.parse_args()
    selection = json.loads(args.selection_lock.read_text())
    if selection.get("status") != "VALIDATION_SELECTION_LOCKED_BEFORE_FRESH_TEST":
        raise RuntimeError("selection is not locked")
    api = HfApi()
    who = api.whoami()["name"]
    if who != args.namespace:
        raise RuntimeError(f"HF account mismatch: expected {args.namespace}, got {who}")
    output = args.run_dir / "hf_uploads.jsonl"
    complete = {}
    if output.is_file():
        complete = {row["method"]: row for row in
                    (json.loads(line) for line in output.read_text().splitlines() if line.strip())
                    if row.get("verified") is True}
    for method, selection_row in sorted(selection["selected_by_method"].items()):
        if method in complete:
            continue
        candidate = selection_row["candidate_id"]
        folder = args.fair_root / "sweep/candidates" / candidate
        if not model_present(folder):
            raise RuntimeError(f"model missing for {candidate}")
        repo_id = f"{args.namespace}/ronpo-qwen3-8b-fair-{method.replace('_', '-')}-s42"
        api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
        card_dir = args.run_dir / "hf_model_cards"
        card_dir.mkdir(parents=True, exist_ok=True)
        card = card_dir / f"{method}.md"
        card.write_text(
            "---\n"
            "base_model: Qwen/Qwen3-8B\n"
            "library_name: transformers\n"
            "tags:\n"
            "- preference-optimization\n"
            "- ronpo\n"
            "- qwen3\n"
            "---\n\n"
            f"# Qwen3-8B fair-demo checkpoint: {method}\n\n"
            f"Validation-selected candidate: `{candidate}`. Exact evaluator, selection, sweep, "
            "and training metadata are stored under `experiment/`.\n",
            encoding="utf-8",
        )
        commit = api.upload_folder(
            repo_id=repo_id, repo_type="model", folder_path=str(folder),
            ignore_patterns=["README.md", "checkpoint-*/*", "wandb/*", "optimizer.pt", "scheduler.pt", "rng_state.pth"],
            commit_message=f"Upload fair-demo validation-selected {candidate}",
        )
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo_id,
                        repo_type="model", commit_message="Add valid public model card")
        for source, target in [
            (args.run_dir / "evaluator_lock.json", "experiment/evaluator_lock.json"),
            (args.selection_lock, "experiment/selection_lock.json"),
            (args.run_dir / "sweep/grid.json", "experiment/sweep_grid.json"),
            (folder / "training_status.json", "experiment/training_status.json"),
        ]:
            api.upload_file(path_or_fileobj=str(source), path_in_repo=target, repo_id=repo_id,
                            repo_type="model", commit_message="Add reproducibility metadata")
        info = api.repo_info(repo_id=repo_id, repo_type="model")
        files = set(api.list_repo_files(repo_id=repo_id, repo_type="model", revision=info.sha))
        has_weight = "model.safetensors" in files or "model.safetensors.index.json" in files
        AutoConfig.from_pretrained(repo_id, revision=info.sha)
        verified = bool(not info.private and "config.json" in files and has_weight)
        row = {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
               "method": method, "candidate_id": candidate, "repo_id": repo_id,
               "url": f"https://huggingface.co/{repo_id}", "revision": info.sha,
               "public": not info.private, "minimal_config_reload": True, "verified": verified,
               "spent_sealed_split_touched": False}
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        if not verified:
            raise RuntimeError(f"public upload verification failed for {repo_id}")


if __name__ == "__main__":
    main()
