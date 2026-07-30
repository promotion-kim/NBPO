#!/usr/bin/env python3
"""Freeze the validation-only checkpoint set for completed RONPO rounds."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def model_complete(path: Path) -> bool:
    return ((path / "model.safetensors").is_file()
            or (path / "model.safetensors.index.json").is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", action="append", required=True,
                        help="round_id=/absolute/round/work")
    parser.add_argument("--grid", action="append", required=True,
                        help="round_id=/path/to/frozen_grid.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-selection-split",
        default="existing prompt-disjoint 128-prompt validation",
    )
    parser.add_argument(
        "--checkpoint-selection-metric",
        default="mean_prompt_worst_standardized_delta",
    )
    args = parser.parse_args()
    rounds = dict(item.split("=", 1) for item in args.round)
    grids = {name: Path(path) for name, path in (item.split("=", 1) for item in args.grid)}
    if set(rounds) != set(grids):
        raise RuntimeError("round and grid identifiers differ")
    rows = []
    grid_hashes = {}
    for round_id in sorted(rounds):
        root = Path(rounds[round_id])
        manifest = json.loads((root / "training_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or manifest.get("failures"):
            raise RuntimeError(f"training round {round_id} is not cleanly complete")
        grid = json.loads(grids[round_id].read_text(encoding="utf-8"))
        grid_hashes[round_id] = sha256(grids[round_id])
        candidate_configs = {row["id"]: row for row in grid["candidates"]}
        for candidate_id in manifest["completed"]:
            output = root / "candidates" / candidate_id
            status = json.loads((output / "training_status.json").read_text(encoding="utf-8"))
            if status.get("status") != "completed" or status.get("measured_step") != 900:
                raise RuntimeError(f"incomplete candidate {candidate_id}")
            checkpoints = {}
            for checkpoint in sorted(output.glob("checkpoint-*")):
                try:
                    step = int(checkpoint.name.rsplit("-", 1)[1])
                except ValueError:
                    continue
                if model_complete(checkpoint):
                    checkpoints[step] = checkpoint
            # The final output is the authoritative step-900 artifact.  Earlier
            # checkpoints remain eligible for the preregistered early-stop rule.
            if not model_complete(output):
                raise RuntimeError(f"final model missing for {candidate_id}")
            checkpoints[900] = output
            expected = list(range(100, 901, 100))
            missing = [step for step in expected if step not in checkpoints]
            if missing:
                raise RuntimeError(f"{candidate_id} is missing saved checkpoints {missing}")
            for step in expected:
                rows.append({
                    "model_id": f"{candidate_id}__s{step}",
                    "candidate_id": candidate_id,
                    "round": round_id,
                    "step": step,
                    "model_path": str(checkpoints[step]),
                    "wandb_run_id": status["wandb_run_id"],
                    "wandb_url": status["wandb_url"],
                    "candidate_config": candidate_configs[candidate_id],
                })
    output = {
        "status": "frozen_before_validation_decode_and_ranking",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "checkpoint_selection_split": args.checkpoint_selection_split,
        "checkpoint_selection_metric": args.checkpoint_selection_metric,
        "panel_judgments_used_for_selection": False,
        "grid_sha256": grid_hashes,
        "models": rows,
        "model_count": len(rows),
        "spent_sealed_split_touched": False,
    }
    atomic_json(args.output, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
