#!/usr/bin/env python3
"""Run two frozen seed-42 RONPO recipes only when validation says a sweep is needed."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import yaml

from scripts.revision.flagship import train_flagship as tf


CANDIDATES = {
    "alpha075_anchor005_lr5e8": {
        "learning_rate": 5.0e-8, "ronpo_alpha": 0.75, "ronpo_tau": 0.05,
        "reference_anchor_weight": 0.05, "preference_sft_weight": 0.005,
    },
    "alpha050_anchor0035_lr7p5e8": {
        "learning_rate": 7.5e-8, "ronpo_alpha": 0.50, "ronpo_tau": 0.05,
        "reference_anchor_weight": 0.035, "preference_sft_weight": 0.0035,
    },
}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def run_id(name: str) -> str:
    return hashlib.sha256(f"aaai27-p3-seed42-v1|{name}".encode()).hexdigest()[:12]


def finite_model(output: Path) -> bool:
    return tf.model_complete(output) and tf.metrics_finite(output)[0]


def ifeval_score(output: Path) -> float | None:
    found = []
    for path in output.rglob("*.json") if output.exists() else ():
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        entry = payload.get("results", {}).get("ifeval") if isinstance(payload, dict) else None
        if not isinstance(entry, dict):
            continue
        values = [
            float(value) for key, value in entry.items()
            if key.split(",", 1)[0] == "prompt_level_strict_acc"
            and isinstance(value, (int, float))
        ]
        if len(values) == 1:
            found.append(values[0] * 100.0)
    return found[0] if len(found) == 1 else None


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--decode-python", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--stop-at", required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / "logs").mkdir(exist_ok=True)
    status_path = args.work / "status.json"
    validation_work = args.root / "eval/p1_validation_reward_seed42"
    stop_at = datetime.fromisoformat(args.stop_at).timestamp()

    while time.time() < stop_at:
        source_status = validation_work / "status.json"
        if source_status.exists():
            source = json.loads(source_status.read_text())
            if source.get("status") == "completed":
                break
            if source.get("status") in {"failed", "deadline_reached"}:
                atomic_json(status_path, {"status": "blocked", "validation_status": source,
                                          "p1_sealed_test_opened": False})
                return
        atomic_json(status_path, {"status": "waiting", "stage": "validation_rank",
                                  "p1_sealed_test_opened": False})
        time.sleep(30)
    else:
        atomic_json(status_path, {"status": "deadline_reached", "stage": "validation_rank",
                                  "p1_sealed_test_opened": False})
        return

    if not source.get("needs_seed42_sweep"):
        atomic_json(status_path, {"status": "not_needed", "validation_rank": 1,
                                  "p1_sealed_test_opened": False})
        return

    frozen = json.loads(args.protocol.read_text())
    if (frozen.get("frozen_before_validation_reward_result") is not True
            or frozen.get("seed") != 42 or frozen.get("schema_version") != 2):
        raise RuntimeError("invalid P3 frozen sweep protocol")
    expected = {
        name: {
            "learning_rate": values["learning_rate"],
            "ronpo_alpha": values["ronpo_alpha"],
            "reference_anchor_weight": values["reference_anchor_weight"],
            "preference_sft_weight": values["preference_sft_weight"],
        }
        for name, values in CANDIDATES.items()
    }
    if frozen.get("candidates") != expected:
        raise RuntimeError("code/protocol P3 candidate mismatch")
    (args.work / "sweep_protocol.json").write_text(args.protocol.read_text())

    running = {}
    outputs = {}
    for gpu, (name, overrides) in zip((0, 2), CANDIDATES.items()):
        output = args.work / "candidates" / name
        outputs[name] = output
        output.mkdir(parents=True, exist_ok=True)
        config = tf.unified_config(
            args.base_model, str(args.root / "precomputed/ronpo_full_expect"),
            output, "full", "ronpo_full_expect", 42, 1,
        )
        config.update(overrides)
        config["run_name"] = f"aaai27-p3-ronpo-{name}-s42"
        config_path = output / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        identifier = run_id(name)
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(args.project),
            "HF_HOME": str(args.root / "cache/huggingface"),
            "HF_HUB_CACHE": str(args.root / "cache/huggingface/hub"),
            "WANDB_MODE": "online", "WANDB_ENTITY": "promotion-kim", "WANDB_PROJECT": "mnpo",
            "WANDB_RUN_GROUP": "ronpo-aaai27-p3-seed42-sweep", "WANDB_RUN_ID": identifier,
            "WANDB_NAME": f"aaai27-p3-ronpo-{name}-s42", "WANDB_RESUME": "allow",
            "MNPO_DISABLE_CUDNN_SDPA": "1", "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        })
        log = (args.work / "logs" / f"train_{name}.log").open("a")
        command = [
            args.python, "-m", "accelerate.commands.launch",
            "--config_file", str(args.project / "accelerate_configs/single_gpu.yaml"),
            "--num_processes=1", "-m", "mnpo_scripts.run_mnpo", str(config_path),
        ]
        process = subprocess.Popen(command, cwd=args.project, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        running[gpu] = (name, process, log, identifier)

    eligible = {}
    failed = []
    while running and time.time() < stop_at:
        atomic_json(status_path, {
            "status": "running", "stage": "sweep_training_s3",
            "running": [{"gpu": gpu, "candidate": value[0], "pid": value[1].pid,
                         "wandb_run_id": value[3]} for gpu, value in running.items()],
            "eligible": eligible, "failed": failed, "p1_sealed_test_opened": False,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        })
        time.sleep(15)
        for gpu, (name, process, log, identifier) in list(running.items()):
            rc = process.poll()
            if rc is None:
                continue
            log.close()
            output = outputs[name]
            if rc != 0 or not finite_model(output):
                failed.append({"candidate": name, "stage": "training", "returncode": rc})
                del running[gpu]
                continue
            namespace = SimpleNamespace(root=args.root, project=args.project, python=args.python)
            job = SimpleNamespace(output_dir=output, gpu=gpu)
            passed, reason = tf.run_stability_gate(namespace, job)
            gate = json.loads((output / "stability/gate.json").read_text()) if (output / "stability/gate.json").exists() else {}
            metadata = {
                "candidate": name, "seed": 42, "wandb_run_id": identifier,
                "wandb_url": f"https://wandb.ai/promotion-kim/mnpo/runs/{identifier}",
                "config": str(output / "config.yaml"), "s3_passed": passed,
                "s3_reason": reason, "s3_gate": gate,
            }
            atomic_json(output / "candidate_status.json", metadata)
            if passed:
                eligible[name] = metadata
            else:
                failed.append({"candidate": name, "stage": "S3", "reason": reason,
                               "gate": str(output / "stability/gate.json")})
            del running[gpu]

    if running:
        atomic_json(status_path, {"status": "deadline_reached", "stage": "sweep_training_s3",
                                  "p1_sealed_test_opened": False})
        return
    if not eligible:
        current_manifest = json.loads((validation_work / "model_manifest.json").read_text())
        current_rows = json.loads((validation_work / "results/ranked_validation_summary.json").read_text())["ranked"]
        current = next(row for row in current_rows if row["model"] == "ronpo_full_expect")
        selection = {
            "status": "selected", "selected_model_name": "ronpo_full_expect",
            "selected_model": current_manifest["models"]["ronpo_full_expect"],
            "selected_local_path": None,
            "validation_rank": current["validation_worst_objective_rank"],
            "validation_score": current["mean_primary_prompt_worst_norm_score"],
            "selection_split": "non-sealed validation",
            "selection_metric": "mean_primary_prompt_worst_norm_score",
            "reason": "All P3 candidates failed the frozen S3 gate; fail-closed fallback to the existing eligible model.",
            "p1_sealed_test_opened": False,
        }
        atomic_json(validation_work / "final_model_selection.json", selection)
        atomic_json(status_path, {"status": "completed_no_eligible_candidate", "failed": failed,
                                  "selection": selection, "p1_sealed_test_opened": False})
        return

    # Candidate S3 generations are stability evidence, not model-selection
    # generations.  Re-decode every eligible candidate with the exact same
    # deterministic vLLM protocol used by the existing methods.
    decode_running = {}
    decode_failures = []
    pending_decode = list(eligible)
    decode_attempts = {}
    while (pending_decode or decode_running) and time.time() < stop_at:
        for gpu, (name, process, handle) in list(decode_running.items()):
            rc = process.poll()
            if rc is None:
                continue
            handle.close()
            output = args.work / "validation_generations" / name / "output_42.json"
            try:
                complete = isinstance(json.loads(output.read_text()), list) and len(json.loads(output.read_text())) == 128
            except (OSError, json.JSONDecodeError):
                complete = False
            if not (rc == 0 and complete):
                decode_failures.append({"candidate": name, "returncode": rc,
                                        "attempt": decode_attempts[name]})
            del decode_running[gpu]
        for gpu in (0, 2):
            if gpu in decode_running or not pending_decode:
                continue
            name = pending_decode.pop(0)
            output_dir = args.work / "validation_generations" / name
            output_dir.mkdir(parents=True, exist_ok=True)
            decode_attempts[name] = decode_attempts.get(name, 0) + 1
            handle = (args.work / "logs" / f"validation_decode_{name}.log").open("a")
            command = [
                args.decode_python,
                str(args.project / "scripts/revision/flagship/decode_vllm_non_thinking.py"),
                "--data-dir", str(args.root / "data/pool_validation.jsonl"),
                "--model", str(outputs[name]), "--output-dir", str(output_dir),
                "--seed", "42", "--temperature", "0.7", "--top-p", "0.9",
                "--max-new-tokens", "2048", "--max-prompts", "128",
            ]
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "TORCH_CUDNN_SDPA_ENABLED": "0",
                        "TOKENIZERS_PARALLELISM": "false",
                        "HF_HOME": str(args.root / "cache/huggingface"),
                        "HF_HUB_CACHE": str(args.root / "cache/huggingface/hub")})
            process = subprocess.Popen(command, cwd=args.project, env=env,
                                       stdout=handle, stderr=subprocess.STDOUT)
            decode_running[gpu] = (name, process, handle)
        atomic_json(status_path, {
            "status": "running", "stage": "sweep_validation_vllm_decode",
            "pending": pending_decode,
            "running": [{"gpu": gpu, "candidate": value[0], "pid": value[1].pid}
                        for gpu, value in decode_running.items()],
            "failures": decode_failures, "p1_sealed_test_opened": False,
        })
        time.sleep(15)
    if pending_decode or decode_running or decode_failures:
        atomic_json(status_path, {"status": "failed", "stage": "sweep_validation_vllm_decode",
                                  "pending": pending_decode, "failures": decode_failures,
                                  "p1_sealed_test_opened": False})
        return

    merged = args.work / "validation_merged_with_sweep.json"
    generations = []
    current_manifest = json.loads((validation_work / "model_manifest.json").read_text())
    for method in current_manifest["models"]:
        generations.append(
            f"{method}={validation_work / 'generations' / method / 'output_42.json'}"
        )
    for name in eligible:
        generations.append(
            f"ronpo_sweep_{name}={args.work / 'validation_generations' / name / 'output_42.json'}"
        )
    command = [args.python, "-m", "mnpo_scripts.merge_model_generations",
               "--generations", *generations, "--output_file", str(merged)]
    with (args.work / "logs/merge.log").open("a") as handle:
        subprocess.run(command, cwd=args.project, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)

    score_dir = args.work / "validation_scores"
    with (args.work / "logs/score.log").open("a") as handle:
        subprocess.run([
            args.python, str(args.project / "scripts/revision/flagship/score_armo_primary_heads_sharded.py"),
            "--python", args.python,
            "--input-file", str(merged), "--output-dir", str(score_dir),
            "--cache-dir", str(args.root / "cache/huggingface/hub"),
            "--gpu-ids", "0", "2",
            "--batch-size", "8", "--sample-batch-size", "4", "--local-files-only",
        ], cwd=args.project, stdout=handle, stderr=subprocess.STDOUT, check=True)
    result_dir = args.work / "validation_results"
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
    ranked = sorted(rows, key=lambda row: (-float(row["mean_primary_prompt_worst_norm_score"]), row["model"]))
    values = [float(row["mean_primary_prompt_worst_norm_score"]) for row in ranked]
    for row, value in zip(ranked, values):
        row["validation_worst_objective_rank"] = 1 + sum(other > value + 1e-12 for other in values)
    atomic_json(result_dir / "ranked_validation_summary.json", {"ranked": ranked,
        "selection_split": "non-sealed validation", "p1_sealed_test_opened": False})
    subprocess.run([
        args.python, str(args.project / "scripts/revision/flagship/log_reward_results_wandb.py"),
        "--summary", str(result_dir / "ranked_validation_summary.json"),
        "--stage", "p3-sweep-validation-reward", "--output", str(args.work / "wandb_eval_run.json"),
    ], cwd=args.project, check=True)

    # Measure IFEval for each S3-passing candidate with the same frozen P2
    # lm-eval/vLLM settings before selection.  Existing-method scores come
    # from the already-complete P2 artifact.
    pending_ifeval = list(eligible)
    ifeval_running = {}
    ifeval_failures = []
    ifeval_attempts = {}
    while (pending_ifeval or ifeval_running) and time.time() < stop_at:
        for gpu, (name, process, handle) in list(ifeval_running.items()):
            rc = process.poll()
            if rc is None:
                continue
            handle.close()
            measured = ifeval_score(args.work / "ifeval" / name)
            if rc != 0 or measured is None:
                ifeval_failures.append({"candidate": name, "returncode": rc,
                                        "attempt": ifeval_attempts[name]})
            del ifeval_running[gpu]
        for gpu in (0, 2):
            if gpu in ifeval_running or not pending_ifeval:
                continue
            name = pending_ifeval.pop(0)
            output = args.work / "ifeval" / name
            ifeval_attempts[name] = ifeval_attempts.get(name, 0) + 1
            identifier = run_id(f"ifeval-{name}")
            command = [
                args.decode_python, "-m", "lm_eval", "run", "--model", "vllm",
                "--model_args", ",".join([
                    f"pretrained={outputs[name]}", "dtype=bfloat16",
                    "gpu_memory_utilization=0.88", "max_model_len=32768",
                    "enable_thinking=False",
                ]),
                "--tasks", "ifeval", "--num_fewshot", "0", "--batch_size", "auto",
                "--max_batch_size", "256", "--apply_chat_template",
                "--seed", "42,42,42,42", "--cache_requests", "true", "--show_config",
                "--output_path", str(output),
                "--wandb_args", "entity=promotion-kim", "project=mnpo", f"id={identifier}",
                f"name=aaai27-p3-ifeval-{name}-s42", "group=ronpo-aaai27-p3-seed42-sweep",
                "job_type=lm_eval", "resume=allow",
                "--wandb_config_args", "flagship_stage=P3_candidate_IFEval",
                f"model_name=ronpo_sweep_{name}", "task_group=ifeval", "num_fewshot=0",
                "enable_thinking=False",
            ]
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": str(gpu),
                        "HF_HOME": str(args.root / "cache/huggingface"),
                        "HF_HUB_CACHE": str(args.root / "cache/huggingface/hub"),
                        "HF_DATASETS_CACHE": str(args.root / "cache/huggingface/datasets"),
                        "WANDB_MODE": "online", "WANDB_ENTITY": "promotion-kim",
                        "WANDB_PROJECT": "mnpo", "TOKENIZERS_PARALLELISM": "false",
                        "TORCH_CUDNN_SDPA_ENABLED": "0",
                        "VLLM_HOST_IP": "127.0.0.1", "VLLM_PORT": str(62000 + gpu * 20),
                        "VLLM_DP_MASTER_PORT": str(62001 + gpu * 20),
                        "MASTER_PORT": str(62002 + gpu * 20)})
            handle = (args.work / "logs" / f"ifeval_{name}.log").open("a")
            process = subprocess.Popen(command, cwd=args.project, env=env,
                                       stdout=handle, stderr=subprocess.STDOUT)
            ifeval_running[gpu] = (name, process, handle)
        atomic_json(status_path, {"status": "running", "stage": "sweep_candidate_ifeval",
                                  "pending": pending_ifeval,
                                  "running": [{"gpu": gpu, "candidate": value[0], "pid": value[1].pid}
                                              for gpu, value in ifeval_running.items()],
                                  "failures": ifeval_failures, "p1_sealed_test_opened": False})
        time.sleep(20)
    if pending_ifeval or ifeval_running or ifeval_failures:
        atomic_json(status_path, {"status": "failed", "stage": "sweep_candidate_ifeval",
                                  "pending": pending_ifeval, "failures": ifeval_failures,
                                  "p1_sealed_test_opened": False})
        return

    base_ifeval_payload = json.loads(
        (args.root / "eval/p2_ifeval_seed42/results/ifeval_seed42.json").read_text()
    )
    baseline_ifeval = {
        row["method"]: float(row["ifeval_prompt_strict_percent"])
        for row in base_ifeval_payload["rows"] if row["method"] != "ronpo_full_expect"
    }
    base_ifeval = baseline_ifeval["base"]
    current_ifeval = float(next(
        row["ifeval_prompt_strict_percent"] for row in base_ifeval_payload["rows"]
        if row["method"] == "ronpo_full_expect"
    ))
    baseline_reward_values = [
        float(row["mean_primary_prompt_worst_norm_score"]) for row in ranked
        if row["model"] != "ronpo_full_expect" and not row["model"].startswith("ronpo_sweep_")
    ]
    options = []
    for row in ranked:
        model_name = row["model"]
        if model_name != "ronpo_full_expect" and not model_name.startswith("ronpo_sweep_"):
            continue
        if model_name == "ronpo_full_expect":
            ivalue = current_ifeval
        else:
            candidate_name = model_name.removeprefix("ronpo_sweep_")
            ivalue = ifeval_score(args.work / "ifeval" / candidate_name)
            if ivalue is None:
                raise RuntimeError(f"missing candidate IFEval score: {candidate_name}")
        reward_value = float(row["mean_primary_prompt_worst_norm_score"])
        option = {
            "model": model_name,
            "validation_worst_objective_score": reward_value,
            "prospective_validation_worst_rank": 1 + sum(
                value > reward_value + 1e-12 for value in baseline_reward_values
            ),
            "ifeval_percent": ivalue,
            "prospective_ifeval_rank": 1 + sum(
                value > ivalue + 1e-12 for value in baseline_ifeval.values()
            ),
            "ifeval_no_regression_vs_base": ivalue + 1e-12 >= base_ifeval,
        }
        options.append(option)
    preferred = [option for option in options
                 if option["prospective_validation_worst_rank"] == 1
                 and option["ifeval_no_regression_vs_base"]]
    if preferred:
        chosen_option = min(preferred, key=lambda option: (
            option["prospective_ifeval_rank"], -option["ifeval_percent"],
            -option["validation_worst_objective_score"], option["model"],
        ))
    else:
        chosen_option = min(options, key=lambda option: (
            -option["validation_worst_objective_score"], option["prospective_ifeval_rank"],
            -option["ifeval_percent"], option["model"],
        ))
    chosen = next(row for row in ranked if row["model"] == chosen_option["model"])
    atomic_json(args.work / "candidate_selection_metrics.json", {
        "protocol": str(args.protocol), "options": options,
        "selected": chosen_option, "p1_sealed_test_opened": False,
    })
    if chosen["model"] == "ronpo_full_expect":
        selected_model = current_manifest["models"]["ronpo_full_expect"]
        selected_path = None
    else:
        candidate_name = chosen["model"].removeprefix("ronpo_sweep_")
        selected_path = outputs[candidate_name]
        repo_id = f"promotion/qwen3-8b-aaai27-p3-ronpo-{candidate_name}-s42"
        ledger = args.work / "hf_uploads.jsonl"
        upload_log = args.work / "logs/upload_selected.log"
        with upload_log.open("a") as handle:
            subprocess.run([
                args.python, str(args.project / "scripts/revision/upload_checkpoint_to_hf.py"),
                "--local-path", str(selected_path), "--repo-id", repo_id,
                "--method", f"ronpo_full_expect_sweep_{candidate_name}",
                "--base-model", "Qwen/Qwen3-8B", "--seed", "42",
                "--notes", "Seed-42 P3 candidate selected only on non-sealed validation after strict S3 pass.",
                "--ledger", str(ledger),
            ], cwd=args.project, stdout=handle, stderr=subprocess.STDOUT, check=True)
        upload_row = json.loads(ledger.read_text().splitlines()[-1])
        selected_model = {"repo_id": upload_row["repo_id"],
                          "revision": upload_row["upload_commit"].rsplit("/", 1)[-1],
                          "upload_commit": upload_row["upload_commit"], "seed": 42}
    selection = {
        "status": "selected", "selected_model_name": chosen["model"],
        "selected_model": selected_model, "selected_local_path": str(selected_path) if selected_path else None,
        "validation_rank": chosen["validation_worst_objective_rank"],
        "validation_score": chosen["mean_primary_prompt_worst_norm_score"],
        "prospective_validation_worst_rank": chosen_option["prospective_validation_worst_rank"],
        "ifeval_percent": chosen_option["ifeval_percent"],
        "prospective_ifeval_rank": chosen_option["prospective_ifeval_rank"],
        "ifeval_no_regression_vs_base": chosen_option["ifeval_no_regression_vs_base"],
        "selection_split": "non-sealed validation",
        "selection_metric": "mean_primary_prompt_worst_norm_score",
        "p1_sealed_test_opened": False,
    }
    atomic_json(validation_work / "final_model_selection.json", selection)
    atomic_json(status_path, {"status": "completed", "stage": "validation_selection",
                              "selection": selection, "eligible": eligible, "failed": failed,
                              "p1_sealed_test_opened": False})


if __name__ == "__main__":
    main()
