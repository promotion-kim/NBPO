#!/usr/bin/env python3
"""Single-open sealed P1 reward evaluation after final RONPO model selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
_RUNTIME_STATUS: Path | None = None
_RUNTIME_STAGE = "preflight"
_SEALED_OPENED = False


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def ledger_models(root: Path, base_revision: str) -> dict[str, dict]:
    result = {"base": {"repo_id": "Qwen/Qwen3-8B", "revision": base_revision, "seed": None}}
    for line in (root / "hf_uploads.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        method = str(row.get("method"))
        if method not in METHODS or str(row.get("seed")) != "42" or row.get("verified") is not True:
            continue
        commit = str(row.get("upload_commit", ""))
        if "/commit/" in commit:
            result[method] = {"repo_id": row["repo_id"], "revision": commit.rsplit("/", 1)[-1],
                              "upload_commit": commit, "seed": 42}
    return result


def frozen_models_tsv(path: Path, base_revision: str) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    result = {}
    for row in rows:
        name = row.get("name")
        if name not in METHODS:
            continue
        result[name] = {
            "repo_id": row["model"], "revision": row["revision"],
            "upload_commit": row.get("upload_commit") or None,
            "seed": int(row["seed"]) if row.get("seed") else None,
        }
    if result.get("base", {}).get("revision") != base_revision:
        raise RuntimeError("models.tsv base revision does not match --base-revision")
    return result


def complete(path: Path, expected: int) -> bool:
    try:
        rows = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(rows, list) and len(rows) == expected


def load_selection_lock(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError("selection_lock.json is absent; sealed evaluation refused")
    lock = json.loads(path.read_text())
    required = {
        "status": "locked",
        "selection_split": "non-sealed validation",
        "sealed_data_consulted_for_selection": False,
        "p1_sealed_test_opened": False,
    }
    for key, expected in required.items():
        if lock.get(key) != expected:
            raise RuntimeError(f"invalid selection lock field {key!r}: {lock.get(key)!r}")
    if not lock.get("selected_ronpo_variant") or not isinstance(lock.get("selected_model"), dict):
        raise RuntimeError("selection lock lacks the selected RONPO model identity")
    for key in ("repo_id", "revision"):
        if not lock["selected_model"].get(key):
            raise RuntimeError(f"selection lock selected_model lacks {key}")
    return lock


def run_stability_gates(args: argparse.Namespace, methods: list[str]) -> dict[str, dict]:
    gate_dir = args.work / "stability_gates"
    gate_dir.mkdir(exist_ok=True)
    base = args.work / "generations/base/output_42.json"
    gates: dict[str, dict] = {}
    for method in methods:
        output = gate_dir / f"{method}.json"
        command = [
            args.python, str(args.project / "scripts/revision/flagship/stability_gate.py"),
            "--base", str(base),
            "--candidate", str(args.work / "generations" / method / "output_42.json"),
            "--output", str(output),
            "--min-length-ratio", "0.33",
            "--max-length-ratio", "2.0",
            "--max-repeat-run", "20",
            "--expected-records", str(args.expected_prompts),
        ]
        with (args.work / "logs" / f"stability_{method}.log").open("a") as handle:
            completed = subprocess.run(command, cwd=args.project, stdout=handle,
                                       stderr=subprocess.STDOUT, check=False)
        try:
            gate = json.loads(output.read_text())
        except (OSError, json.JSONDecodeError):
            gate = {"status": "failed", "passed": False,
                    "error": "stability gate artifact missing or invalid",
                    "returncode": completed.returncode}
        gates[method] = gate
    atomic_json(gate_dir / "summary.json", {
        "fail_closed": True,
        "thresholds": {"records": args.expected_prompts, "empty_count": 0,
                       "think_leak_count": 0, "min_length_ratio": 0.33,
                       "max_length_ratio": 2.0, "max_repeat_run": 20},
        "models": gates,
        "all_passed": all(gate.get("passed") is True for gate in gates.values()),
    })
    return gates


def report_markdown(result_dir: Path, ranked: list[dict], per_objective_path: Path,
                    gates: dict[str, dict], selection: dict, digest: str,
                    wandb: dict, prompt_count: int, models: dict[str, dict]) -> None:
    per_objective: dict[str, dict[str, dict]] = {}
    with per_objective_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            per_objective.setdefault(row["model"], {})[row["objective"]] = row
    lines = [
        "# P1 sealed reward report",
        "",
        f"Selected RONPO variant (locked before sealed access): `{selection['selected_ronpo_variant']}`.",
        "",
        "| Rank | Model | Worst (95% CI) | Avg | Win vs base | Helpfulness | Safety | Conciseness | Stability |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in ranked:
        model = row["model"]
        objectives = per_objective.get(model, {})
        low = row["mean_primary_prompt_worst_norm_score_ci95_low"]
        high = row["mean_primary_prompt_worst_norm_score_ci95_high"]
        win = row.get("mean_win_rate_vs_baseline")
        win_text = "--" if win is None or not math.isfinite(float(win)) else f"{100*float(win):.2f}%"
        values = [objectives.get(name, {}).get("mean_prompt_norm_score")
                  for name in ("helpfulness", "safety", "conciseness")]
        obj_text = [("--" if value in (None, "") else f"{float(value):.4f}") for value in values]
        lines.append(
            f"| {row['worst_objective_rank']} | {model} | "
            f"{float(row['mean_primary_prompt_worst_norm_score']):.4f} "
            f"[{float(low):.4f}, {float(high):.4f}] | "
            f"{float(row['mean_primary_prompt_avg_norm_score']):.4f} | {win_text} | "
            f"{obj_text[0]} | {obj_text[1]} | {obj_text[2]} | "
            f"{gates[model].get('status', 'failed')} |"
        )
    lines.extend([
        "", "## Provenance", "",
        f"- Prompt count: {prompt_count}",
        f"- Sealed prompt SHA-256: `{digest}`",
        "- Decode: vLLM; seed 42; temperature 0.7; top-p 0.9; max_new_tokens 2048; "
        "chat template; enable_thinking=false; bfloat16.",
        "- Reward model: `RLHFlow/ArmoRM-Llama3-8B-v0.1`; heads "
        "`ultrafeedback-helpfulness`, `beavertails-is_safe`, and negated "
        "`helpsteer-verbosity`.",
        "- Normalization: per-prompt min-max over the evaluated sealed model pool.",
        "- Intervals: 2,000-resample paired prompt bootstrap, seed 42.",
        f"- W&B run ID: `{wandb.get('wandb_run_id', 'unknown')}` "
        f"({wandb.get('wandb_url', 'unknown')})",
        "- Exact model revisions:",
    ])
    for method, model in models.items():
        lines.append(f"  - `{method}`: `{model['repo_id']}@{model['revision']}`")
    lines.append("")
    (result_dir / "SEALED_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def validate_measured_results(result_dir: Path, rows: list[dict], methods: list[str],
                              expected_prompts: int) -> None:
    if len(rows) != len(methods) or {row.get("model") for row in rows} != set(methods):
        raise RuntimeError("sealed model_summary does not contain exactly the frozen model pool")
    required = (
        "mean_primary_prompt_worst_norm_score",
        "mean_primary_prompt_worst_norm_score_ci95_low",
        "mean_primary_prompt_worst_norm_score_ci95_high",
        "mean_primary_prompt_avg_norm_score",
        "mean_primary_prompt_avg_norm_score_ci95_low",
        "mean_primary_prompt_avg_norm_score_ci95_high",
    )
    for row in rows:
        if int(row.get("num_prompts", 0)) != expected_prompts:
            raise RuntimeError(f"wrong prompt count in measured summary for {row.get('model')}")
        for field in required:
            if not math.isfinite(float(row[field])):
                raise RuntimeError(f"non-finite measured field {field} for {row.get('model')}")
    per_objective = list(csv.DictReader(
        (result_dir / "per_objective_scores.csv").open(newline="", encoding="utf-8")
    ))
    expected_keys = {(model, objective) for model in methods
                     for objective in ("helpfulness", "safety", "conciseness")}
    actual_keys = {(row.get("model"), row.get("objective")) for row in per_objective}
    if len(per_objective) != len(expected_keys) or actual_keys != expected_keys:
        raise RuntimeError("per-objective CSV is incomplete or duplicated")
    for row in per_objective:
        for field in ("mean_prompt_norm_score", "mean_prompt_norm_score_ci95_low",
                      "mean_prompt_norm_score_ci95_high", "mean_raw_score",
                      "mean_raw_score_ci95_low", "mean_raw_score_ci95_high"):
            if not math.isfinite(float(row[field])):
                raise RuntimeError(f"non-finite per-objective field {field} for {row.get('model')}")


def main() -> None:
    global _RUNTIME_STATUS, _RUNTIME_STAGE, _SEALED_OPENED
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--decode-python", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--sealed-prompts", type=Path, required=True)
    parser.add_argument("--sealed-sha256", required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--models-tsv", type=Path, required=True)
    parser.add_argument("--expected-prompts", type=int, default=604)
    parser.add_argument("--base-revision", required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True); (args.work / "logs").mkdir(exist_ok=True)
    status = args.work / "status.json"
    _RUNTIME_STATUS = status

    selection = load_selection_lock(args.selection_lock)
    if not args.sealed_prompts.is_file():
        raise FileNotFoundError(args.sealed_prompts)
    digest = hashlib.sha256(args.sealed_prompts.read_bytes()).hexdigest()
    if digest != args.sealed_sha256:
        raise RuntimeError(f"sealed prompt SHA-256 mismatch: {digest}")

    models = frozen_models_tsv(args.models_tsv, args.base_revision)
    missing = [method for method in METHODS if method not in models]
    if missing:
        raise RuntimeError(f"verified models missing: {missing}")
    selected = selection["selected_model"]
    selected_name = selection["selected_ronpo_variant"]
    matching = [name for name, model in models.items()
                if model["repo_id"] == selected["repo_id"] and model["revision"] == selected["revision"]]
    if not matching:
        models["ronpo_selected"] = selected
    methods = list(METHODS) + (["ronpo_selected"] if "ronpo_selected" in models else [])
    opened_path = args.work / "sealed_opened.json"
    opened_payload = {
        "opened_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "reason": "Selection was locked on non-sealed validation before the sole P1 sealed evaluation.",
        "sealed_sha256": digest, "selection_lock": selection,
        "selected_model_row": matching[0] if matching else "ronpo_selected",
        "models": models, "single_open_policy": True,
    }
    if opened_path.exists():
        previous = json.loads(opened_path.read_text())
        for key in ("sealed_sha256", "selection_lock", "models"):
            if previous.get(key) != opened_payload.get(key):
                raise RuntimeError("sealed evaluation was already opened with a different frozen manifest")
    else:
        atomic_json(opened_path, opened_payload)
    _SEALED_OPENED = True
    _RUNTIME_STAGE = "sealed_decode"
    atomic_json(status, {"status": "running", "stage": "sealed_decode",
                         "sealed_test_opened": True, "p1_sealed_test_opened": True})

    pending = list(methods); running = {}; attempts = {}; failed = []
    while pending or running:
        for gpu, (method, process, handle) in list(running.items()):
            rc = process.poll()
            if rc is None:
                continue
            handle.close()
            output = args.work / "generations" / method / "output_42.json"
            if rc == 0 and complete(output, args.expected_prompts):
                pass
            elif attempts[method] < 2:
                pending.insert(0, method)
            else:
                failed.append({"method": method, "returncode": rc, "attempt": attempts[method]})
            del running[gpu]
        for gpu in (0, 1, 2, 3):
            if gpu in running or not pending:
                continue
            method = pending.pop(0)
            output_dir = args.work / "generations" / method
            output = output_dir / "output_42.json"
            if complete(output, args.expected_prompts):
                continue
            attempts[method] = attempts.get(method, 0) + 1
            output_dir.mkdir(parents=True, exist_ok=True)
            handle = (args.work / "logs" / f"decode_{method}_a{attempts[method]}.log").open("a")
            model = models[method]
            command = [
                args.decode_python, str(args.project / "scripts/revision/flagship/decode_vllm_non_thinking.py"),
                "--data-dir", str(args.sealed_prompts), "--model", model["repo_id"],
                "--revision", model["revision"], "--output-dir", str(output_dir),
                "--seed", "42", "--temperature", "0.7", "--top-p", "0.9",
                "--max-new-tokens", "2048",
            ]
            env = os.environ.copy(); env.update({"CUDA_VISIBLE_DEVICES": str(gpu),
                "TORCH_CUDNN_SDPA_ENABLED": "0", "TOKENIZERS_PARALLELISM": "false",
                "HF_HOME": str(args.root / "cache/huggingface"),
                "HF_HUB_CACHE": str(args.root / "cache/huggingface/hub")})
            process = subprocess.Popen(command, cwd=args.project, env=env,
                                       stdout=handle, stderr=subprocess.STDOUT)
            running[gpu] = (method, process, handle)
        atomic_json(status, {"status": "running", "stage": "sealed_decode",
            "pending": pending, "running": [{"gpu": gpu, "method": value[0], "pid": value[1].pid}
            for gpu, value in running.items()], "failed": failed,
            "sealed_test_opened": True, "p1_sealed_test_opened": True})
        time.sleep(15)
    if failed:
        atomic_json(status, {"status": "failed", "stage": "sealed_decode",
                             "failed": failed, "sealed_test_opened": True,
                             "p1_sealed_test_opened": True})
        return

    _RUNTIME_STAGE = "sealed_stability_gates"
    atomic_json(status, {"status": "running", "stage": "sealed_stability_gates",
                         "sealed_test_opened": True, "p1_sealed_test_opened": True})
    gates = run_stability_gates(args, methods)
    failed_gates = [method for method, gate in gates.items() if gate.get("passed") is not True]
    if failed_gates:
        atomic_json(status, {"status": "failed", "stage": "sealed_stability_gates",
            "failed_models": failed_gates, "stability_gates": gates,
            "sealed_test_opened": True, "p1_sealed_test_opened": True})
        return

    _RUNTIME_STAGE = "sealed_merge"
    merged = args.work / "merged_generations.json"
    command = [args.python, "-m", "mnpo_scripts.merge_model_generations", "--generations"]
    command.extend([f"{method}={args.work / 'generations' / method / 'output_42.json'}" for method in methods])
    command.extend(["--output_file", str(merged)])
    with (args.work / "logs/merge.log").open("a") as handle:
        subprocess.run(command, cwd=args.project, stdout=handle, stderr=subprocess.STDOUT, check=True)

    _RUNTIME_STAGE = "sealed_armo_scoring"
    score_dir = args.work / "scores"
    atomic_json(status, {"status": "running", "stage": "sealed_armo_scoring",
                         "sealed_test_opened": True, "p1_sealed_test_opened": True})
    with (args.work / "logs/score.log").open("a") as handle:
        subprocess.run([
            args.python, str(args.project / "scripts/revision/flagship/score_armo_primary_heads_sharded.py"),
            "--python", args.python,
            "--input-file", str(merged), "--output-dir", str(score_dir),
            "--cache-dir", str(args.root / "cache/huggingface/hub"),
            "--gpu-ids", "0", "1", "2", "3",
            "--batch-size", "8", "--sample-batch-size", "4", "--local-files-only",
        ], cwd=args.project, stdout=handle, stderr=subprocess.STDOUT, check=True)
    _RUNTIME_STAGE = "sealed_aggregation"
    result_dir = args.work / "results"
    with (args.work / "logs/evaluate.log").open("a") as handle:
        subprocess.run([
            args.python, "-m", "mnpo_scripts.evaluate_multi_objective_models",
            "--scored_files", f"helpfulness={score_dir / 'helpfulness.jsonl'}",
            f"safety={score_dir / 'safety.jsonl'}", f"conciseness={score_dir / 'conciseness.jsonl'}",
            "--output_dir", str(result_dir), "--baseline_model", "base",
            "--primary_objectives", "helpfulness", "safety", "conciseness",
            "--bootstrap_samples", "2000", "--bootstrap_seed", "42",
        ], cwd=args.project, stdout=handle, stderr=subprocess.STDOUT, check=True)
    rows = json.loads((result_dir / "model_summary.json").read_text())
    validate_measured_results(result_dir, rows, methods, args.expected_prompts)
    ranked = sorted(rows, key=lambda row: (-float(row["mean_primary_prompt_worst_norm_score"]), row["model"]))
    values = [float(row["mean_primary_prompt_worst_norm_score"]) for row in ranked]
    for row, value in zip(ranked, values):
        row["worst_objective_rank"] = 1 + sum(other > value + 1e-12 for other in values)
    with (result_dir / "headline_table.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["worst_objective_rank", "model", "mean_primary_prompt_worst_norm_score",
                  "mean_primary_prompt_worst_norm_score_ci95_low",
                  "mean_primary_prompt_worst_norm_score_ci95_high",
                  "mean_primary_prompt_avg_norm_score", "num_prompts"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(ranked)
    atomic_json(result_dir / "ranked_sealed_summary.json", {
        "selection_split": "held-out sealed test",
        "metric": "mean_primary_prompt_worst_norm_score", "ranked": ranked,
        "bootstrap_resamples": 2000, "sealed_sha256": digest,
        "final_selection": selection, "p1_sealed_test_opened": True,
        "stability_gates": {model: gate.get("status") for model, gate in gates.items()},
    })
    _RUNTIME_STAGE = "sealed_wandb_and_report"
    wandb_env = os.environ.copy()
    wandb_env.update({"WANDB_MODE": "online", "WANDB_ENTITY": "promotion-kim",
                      "WANDB_PROJECT": "mnpo"})
    subprocess.run([
        args.python, str(args.project / "scripts/revision/flagship/log_reward_results_wandb.py"),
        "--summary", str(result_dir / "ranked_sealed_summary.json"),
        "--stage", "p1-sealed-reward", "--output", str(args.work / "wandb_run.json"),
    ], cwd=args.project, env=wandb_env, check=True)
    wandb = json.loads((args.work / "wandb_run.json").read_text())
    report_markdown(result_dir, ranked, result_dir / "per_objective_scores.csv",
                    gates, selection, digest, wandb, args.expected_prompts, models)
    selected_row = matching[0] if matching else "ronpo_selected"
    ronpo = next(row for row in ranked if row["model"] == selected_row)
    atomic_json(status, {"status": "completed", "stage": "measured_sealed_results",
        "selected_ronpo_variant": selected_name, "selected_model_row": selected_row,
        "ronpo_worst_objective_rank": ronpo["worst_objective_rank"],
        "ronpo_worst_objective_score": ronpo["mean_primary_prompt_worst_norm_score"],
        "stability_gates": {model: gate.get("status") for model, gate in gates.items()},
        "sealed_test_opened": True, "p1_sealed_test_opened": True,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds")})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if _RUNTIME_STATUS is not None and _SEALED_OPENED:
            atomic_json(_RUNTIME_STATUS, {
                "status": "failed", "stage": _RUNTIME_STAGE,
                "error": f"{type(error).__name__}: {error}",
                "sealed_test_opened": True, "p1_sealed_test_opened": True,
                "failed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            })
        raise
