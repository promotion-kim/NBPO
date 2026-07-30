#!/usr/bin/env python3
"""Non-sealed seed-42 model-selection evaluation for the frozen ArmoRM triple."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


METHODS = (
    "base", "ronpo_full_expect", "ronpo_k_only", "dpo", "ipo", "simpo",
    "sppo_avg", "inpo_avg", "ht_mnpo_helpfulness", "ht_mnpo_safety",
    "ht_mnpo_conciseness",
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def models(root: Path, base_revision: str) -> dict[str, dict]:
    selected = {
        "base": {"repo_id": "Qwen/Qwen3-8B", "revision": base_revision, "seed": None}
    }
    if not (root / "hf_uploads.jsonl").exists():
        return selected
    for line in (root / "hf_uploads.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        method = str(row.get("method"))
        if method not in METHODS or str(row.get("seed")) != "42" or row.get("verified") is not True:
            continue
        commit = str(row.get("upload_commit", ""))
        revision = commit.rsplit("/", 1)[-1] if "/commit/" in commit else ""
        if revision:
            selected[method] = {
                "repo_id": row["repo_id"], "revision": revision, "seed": 42,
                "upload_commit": commit,
            }
    return selected


def generation_complete(path: Path) -> bool:
    try:
        rows = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(rows, list) and len(rows) == 128


def competition_ranks(rows: list[dict], metric: str, field: str) -> None:
    values = [float(row[metric]) for row in rows]
    for row, value in zip(rows, values):
        row[field] = 1 + sum(other > value + 1e-12 for other in values)


def complete_ifeval(root: Path) -> dict | None:
    path = root / "eval/p2_ifeval_seed42/results/ifeval_seed42.json"
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    measured = {str(row.get("method")): row for row in rows}
    if payload.get("complete") is not True or any(method not in measured for method in METHODS):
        return None
    scores = {method: float(measured[method]["ifeval_prompt_strict_percent"]) for method in METHODS}
    value = scores["ronpo_full_expect"]
    return {
        "source_json": str(path), "scores": scores,
        "ronpo_full_expect_percent": value,
        "base_percent": scores["base"],
        "ronpo_full_expect_rank": 1 + sum(other > value + 1e-12 for other in scores.values()),
        "no_regression_vs_base": value + 1e-12 >= scores["base"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--decode-python", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--stop-at", required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / "logs").mkdir(exist_ok=True)
    status = args.work / "status.json"
    stop_at = datetime.fromisoformat(args.stop_at).timestamp()

    selected = models(args.root, args.base_revision)
    while time.time() < stop_at:
        selected = models(args.root, args.base_revision)
        missing = [method for method in METHODS if method not in selected]
        atomic_json(status, {
            "status": "waiting" if missing else "running", "stage": "verified_hf_models",
            "missing_models": missing, "p1_sealed_test_opened": False,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        })
        if not missing:
            break
        time.sleep(30)
    else:
        atomic_json(status, {"status": "deadline_reached", "stage": "verified_hf_models",
                             "p1_sealed_test_opened": False})
        return

    manifest = {
        "split": "non-sealed validation; first 128 lexicographically sorted prompts",
        "decode": {"backend": "vllm", "seed": 42, "temperature": 0.7, "top_p": 0.9,
                   "max_new_tokens": 2048, "enable_thinking": False},
        "models": {method: selected[method] for method in METHODS},
        "p1_sealed_test_opened": False,
    }
    atomic_json(args.work / "model_manifest.json", manifest)

    pending = [
        method for method in METHODS
        if not generation_complete(args.work / "generations" / method / "output_42.json")
    ]
    running = {}
    failures = []
    attempts = {}
    while (pending or running) and time.time() < stop_at:
        for gpu, (method, process, handle) in list(running.items()):
            rc = process.poll()
            if rc is None:
                continue
            handle.close()
            output = args.work / "generations" / method / "output_42.json"
            if rc == 0 and generation_complete(output):
                pass
            elif attempts[method] < 2:
                pending.insert(0, method)
            else:
                failures.append({"method": method, "attempt": attempts[method], "returncode": rc})
            del running[gpu]

        for gpu in (0, 2):
            if gpu in running or not pending:
                continue
            method = pending.pop(0)
            output_dir = args.work / "generations" / method
            output = output_dir / "output_42.json"
            if generation_complete(output):
                continue
            attempts[method] = attempts.get(method, 0) + 1
            output_dir.mkdir(parents=True, exist_ok=True)
            log = (args.work / "logs" / f"decode_{method}_a{attempts[method]}.log").open("a")
            model = selected[method]
            command = [
                args.decode_python, str(args.project / "scripts/revision/flagship/decode_vllm_non_thinking.py"),
                "--data-dir", str(args.root / "data/pool_validation.jsonl"),
                "--model", model["repo_id"], "--revision", model["revision"],
                "--output-dir", str(output_dir),
                "--seed", "42", "--temperature", "0.7", "--top-p", "0.9",
                "--max-new-tokens", "2048", "--max-prompts", "128",
            ]
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "TORCH_CUDNN_SDPA_ENABLED": "0",
                        "TOKENIZERS_PARALLELISM": "false",
                        "HF_HOME": str(args.root / "cache/huggingface"),
                        "HF_HUB_CACHE": str(args.root / "cache/huggingface/hub")})
            process = subprocess.Popen(command, cwd=args.project, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            running[gpu] = (method, process, log)
        atomic_json(status, {
            "status": "running", "stage": "validation_decode", "pending": pending,
            "running": [{"gpu": gpu, "method": value[0], "pid": value[1].pid}
                        for gpu, value in running.items()],
            "failures": failures, "p1_sealed_test_opened": False,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        })
        time.sleep(15)
    if failures or pending or running:
        atomic_json(status, {"status": "failed", "stage": "validation_decode",
                             "failures": failures, "pending": pending,
                             "p1_sealed_test_opened": False})
        return

    merged = args.work / "merged_generations.json"
    merge_command = [args.python, "-m", "mnpo_scripts.merge_model_generations", "--generations"]
    merge_command.extend([
        f"{method}={args.work / 'generations' / method / 'output_42.json'}"
        for method in METHODS
    ])
    merge_command.extend(["--output_file", str(merged)])
    with (args.work / "logs/merge.log").open("a") as handle:
        subprocess.run(merge_command, cwd=args.project, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)

    atomic_json(status, {"status": "running", "stage": "armo_primary_head_scoring",
                         "p1_sealed_test_opened": False})
    score_dir = args.work / "scores"
    score_command = [
        args.python, str(args.project / "scripts/revision/flagship/score_armo_primary_heads_sharded.py"),
        "--python", args.python,
        "--input-file", str(merged), "--output-dir", str(score_dir),
        "--cache-dir", str(args.root / "cache/huggingface/hub"),
        "--gpu-ids", "0", "2",
        "--batch-size", "8", "--sample-batch-size", "4", "--local-files-only",
    ]
    with (args.work / "logs/score.log").open("a") as handle:
        subprocess.run(score_command, cwd=args.project, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)

    result_dir = args.work / "results"
    evaluate = [
        args.python, "-m", "mnpo_scripts.evaluate_multi_objective_models",
        "--scored_files", f"helpfulness={score_dir / 'helpfulness.jsonl'}",
        f"safety={score_dir / 'safety.jsonl'}",
        f"conciseness={score_dir / 'conciseness.jsonl'}",
        "--output_dir", str(result_dir), "--baseline_model", "base",
        "--primary_objectives", "helpfulness", "safety", "conciseness",
        "--bootstrap_samples", "2000", "--bootstrap_seed", "42",
    ]
    with (args.work / "logs/evaluate.log").open("a") as handle:
        subprocess.run(evaluate, cwd=args.project, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)
    summary = json.loads((result_dir / "model_summary.json").read_text())
    ranked = sorted(summary, key=lambda row: (-float(row["mean_primary_prompt_worst_norm_score"]), row["model"]))
    competition_ranks(ranked, "mean_primary_prompt_worst_norm_score",
                      "validation_worst_objective_rank")
    atomic_json(result_dir / "ranked_validation_summary.json", {
        "selection_split": "non-sealed validation", "metric": "mean_primary_prompt_worst_norm_score",
        "ranked": ranked, "p1_sealed_test_opened": False,
    })
    subprocess.run([
        args.python, str(args.project / "scripts/revision/flagship/log_reward_results_wandb.py"),
        "--summary", str(result_dir / "ranked_validation_summary.json"),
        "--stage", "p1-validation-reward", "--output", str(args.work / "wandb_run.json"),
    ], cwd=args.project, check=True)
    ronpo = next(row for row in ranked if row["model"] == "ronpo_full_expect")

    ifeval = complete_ifeval(args.root)
    while ifeval is None and time.time() < stop_at:
        atomic_json(status, {
            "status": "waiting", "stage": "complete_seed42_ifeval_for_p3_trigger",
            "ronpo_full_expect_reward_validation_rank": ronpo["validation_worst_objective_rank"],
            "p1_sealed_test_opened": False,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        })
        time.sleep(20)
        ifeval = complete_ifeval(args.root)
    if ifeval is None:
        atomic_json(status, {"status": "deadline_reached", "stage": "complete_seed42_ifeval_for_p3_trigger",
                             "p1_sealed_test_opened": False})
        return

    needs_sweep = (
        ronpo["validation_worst_objective_rank"] != 1
        or ifeval["ronpo_full_expect_rank"] > 2
        or not ifeval["no_regression_vs_base"]
    )
    if not needs_sweep:
        atomic_json(args.work / "final_model_selection.json", {
            "status": "selected", "selection_split": "non-sealed validation",
            "selection_metric": "mean_primary_prompt_worst_norm_score",
            "selected_model_name": "ronpo_full_expect",
            "selected_model": selected["ronpo_full_expect"],
            "validation_rank": 1,
            "ifeval": ifeval,
            "reason": "The preregistered RONPO full-expect seed-42 model ranked first on non-sealed validation worst-objective reward, was IFEval top-2, and did not regress below base; no sweep was needed.",
            "p1_sealed_test_opened": False,
        })
    atomic_json(status, {
        "status": "completed", "stage": "measured_validation_selection",
        "ronpo_full_expect_rank": ronpo["validation_worst_objective_rank"],
        "ronpo_full_expect_score": ronpo["mean_primary_prompt_worst_norm_score"],
        "ifeval": ifeval,
        "needs_seed42_sweep": needs_sweep,
        "p1_sealed_test_opened": False,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })


if __name__ == "__main__":
    main()
