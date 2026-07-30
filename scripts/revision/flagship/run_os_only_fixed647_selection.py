#!/usr/bin/env python3
"""Score only full-647 gate-passing OS checkpoints and lock a robust selected step."""

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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def complete(path: Path, expected: int = 647) -> bool:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(rows, list) and len(rows) == expected and all(
        str(row.get("prompt", "")).strip() and str(row.get("generated_text", "")).strip()
        for row in rows
    )


def run(command: list[str], cwd: Path, log: Path, env: dict[str, str] | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gate-summary", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--metric-lock", type=Path, required=True)
    parser.add_argument("--new-metric-lock", type=Path, required=True)
    parser.add_argument("--fixed647", type=Path, required=True)
    parser.add_argument("--base-generation", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--general-rm-cache", type=Path, required=True)
    parser.add_argument("--armo-cache", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    gates4096 = json.loads(args.gate_summary.read_text(encoding="utf-8"))
    if gates4096.get("status") != "completed_fail_closed_before_reward_scoring":
        raise RuntimeError("the reward-blind 4096-token gate set is not locked")
    if gates4096.get("reward_scores_consulted") is not False:
        raise RuntimeError("gate set was not frozen outcome-blind")
    by_id = {row["model_id"]: row for row in manifest["models"]}
    eligible_ids = list(gates4096["eligible_model_ids"])
    if not eligible_ids:
        atomic_json(args.work / "selection_lock.json", {
            "status": "TERMINAL_FAILED_NO_FULL647_GATE_PASSER", "selected": None,
            "gate_summary_sha256": sha256(args.gate_summary),
            "reward_scoring_invoked": False, "spent_sealed_split_touched": False})
        return
    if any(model_id not in by_id for model_id in eligible_ids):
        raise RuntimeError("gate/model manifest mismatch")
    generations = args.work / "generations_2048"
    logs = args.work / "logs"; logs.mkdir(parents=True, exist_ok=True)
    base_output = generations / "base/output_42.json"; base_output.parent.mkdir(parents=True, exist_ok=True)
    if not complete(base_output):
        if not complete(args.base_generation):
            raise RuntimeError("compatible frozen base 647 generation is missing")
        shutil.copy2(args.base_generation, base_output)
        metadata = args.base_generation.parent / "decode_metadata.json"
        if metadata.is_file():
            shutil.copy2(metadata, base_output.parent / "decode_metadata.json")
    queued = [by_id[model_id] for model_id in eligible_ids
              if not complete(generations / model_id / "output_42.json")]
    env_base = os.environ.copy()
    env_base.update({"PYTHONPATH": str(args.project), "HF_HOME": str(args.hf_cache.parent),
                     "HF_HUB_CACHE": str(args.hf_cache), "HF_HUB_OFFLINE": "1",
                     "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                     "TORCH_CUDNN_SDPA_ENABLED": "0", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
    running = {}
    while queued or running:
        for gpu in ("0", "1", "2", "3"):
            if not queued or gpu in running:
                continue
            row = queued.pop(0); model_id = row["model_id"]
            output_dir = generations / model_id; output_dir.mkdir(parents=True, exist_ok=True)
            command = [args.python, "-u", str(args.project / "scripts/revision/flagship/decode_vllm_non_thinking.py"),
                       "--data-dir", str(args.fixed647), "--model", row["model_path"],
                       "--output-dir", str(output_dir), "--seed", "42", "--temperature", "0.7",
                       "--top-p", "0.9", "--max-new-tokens", "2048", "--max-model-len", "8192",
                       "--gpu-memory-utilization", "0.88"]
            env = env_base.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu
            handle = (logs / f"decode_{model_id}.log").open("a", encoding="utf-8")
            process = subprocess.Popen(command, cwd=args.project, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT)
            running[gpu] = (row, process, handle, command)
        atomic_json(args.work / "decode_status.json", {
            "status": "running", "queued": [row["model_id"] for row in queued],
            "running": [{"gpu": gpu, "model_id": value[0]["model_id"], "pid": value[1].pid}
                        for gpu, value in running.items()],
            "spent_sealed_split_touched": False})
        if not running:
            continue
        time.sleep(5)
        for gpu, (row, process, handle, command) in list(running.items()):
            rc = process.poll()
            if rc is None:
                continue
            handle.close()
            if rc or not complete(generations / row["model_id"] / "output_42.json"):
                raise RuntimeError(f"common-protocol decode failed: {row['model_id']}, rc={rc}")
            del running[gpu]

    gate_dir = args.work / "gates_2048"; gate_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_id in eligible_ids:
        output = gate_dir / f"{model_id}.json"
        command = [args.python, str(args.project / "scripts/revision/flagship/stability_gate_corrected.py"),
                   "--base", str(base_output), "--candidate", str(generations / model_id / "output_42.json"),
                   "--output", str(output), "--expected-records", "647", "--min-length-ratio", "0.33",
                   "--max-length-ratio", "2.0", "--max-repeat-run", "20"]
        result = subprocess.run(command, cwd=args.project, capture_output=True, text=True)
        (logs / f"gate2048_{model_id}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode not in {0, 4} or not output.is_file():
            raise RuntimeError(f"2048 gate execution failed: {model_id}")
        payload = json.loads(output.read_text(encoding="utf-8"))
        rows.append({"id": model_id, "method": "ronpo_os", "passed": payload.get("passed") is True,
                     "status": payload.get("status"), "checks": payload.get("checks"),
                     "candidate": payload.get("candidate")})
    eligible2048 = [row["id"] for row in rows if row["passed"]]
    gate2048 = {"status": "completed_fail_closed", "detector": "corrected_nonempty_paired_span_v1",
                "eligible_candidates": eligible2048,
                "failed_candidates": [row["id"] for row in rows if not row["passed"]],
                "rows": rows, "spent_sealed_split_touched": False}
    atomic_json(gate_dir / "summary.json", gate2048)
    if not eligible2048:
        atomic_json(args.work / "selection_lock.json", {
            "status": "TERMINAL_FAILED_NO_COMMON_DECODE_GATE_PASSER", "selected": None,
            "gate4096_sha256": sha256(args.gate_summary), "gate2048_sha256": sha256(gate_dir / "summary.json"),
            "spent_sealed_split_touched": False})
        return
    merged = args.work / "merged_os_checkpoints.json"
    merge = [args.python, "-m", "mnpo_scripts.merge_model_generations", "--generations",
             f"base={base_output}"]
    merge.extend(f"{model_id}={generations / model_id / 'output_42.json'}" for model_id in eligible2048)
    merge.extend(["--output_file", str(merged)])
    run(merge, args.project, logs / "merge.log")
    expanded_grid = {"status": "frozen_expanded_os_checkpoint_grid", "candidates": [],
                     "spent_sealed_split_touched": False}
    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    config_by_id = {row["id"]: row for row in grid["candidates"]}
    for model_id in eligible2048:
        row = by_id[model_id]
        expanded_grid["candidates"].append({"id": model_id, "method": "ronpo_os", "stage": 1,
                                            "source": "os_only_stability_hardening_grid",
                                            "model_path": row["model_path"], "step": row["step"],
                                            "candidate_id": row["candidate_id"],
                                            "config": config_by_id[row["candidate_id"]]})
    expanded_path = args.work / "expanded_checkpoint_grid.json"
    atomic_json(expanded_path, expanded_grid)
    run([args.python, str(args.project / "scripts/revision/flagship/score_stage2_os_localrm.py"),
         "--project", str(args.project), "--python", args.python, "--merged", str(merged),
         "--work", str(args.work), "--general-rm-cache", str(args.general_rm_cache),
         "--armo-cache", str(args.armo_cache), "--expected-prompts", "647"],
        args.project, logs / "score_driver.log")
    results = args.work / "results"
    run([args.python, str(args.project / "scripts/revision/flagship/aggregate_stage2_os_localrm.py"),
         "--merged", str(merged), "--score", f"skywork={args.work / 'scores/skywork.jsonl'}",
         "--score", f"athene={args.work / 'scores/athene.jsonl'}",
         "--score", f"armo={args.work / 'scores/armo.jsonl'}", "--grid", str(expanded_path),
         "--gates", str(gate_dir / "summary.json"), "--metric-lock", str(args.metric_lock),
         "--output-dir", str(results), "--split-name", "fixed_647_os_checkpoint_selection"],
        args.project, logs / "aggregate.log")
    summary = json.loads((results / "model_summary.json").read_text(encoding="utf-8"))
    metric = {row["model"]: row for row in summary["ranked_all_eligible_candidates"] if row["model"] != "base"}
    robust = [model_id for model_id in gates4096["robust_neighbor_model_ids"] if model_id in eligible2048]
    pool = robust if robust else eligible2048
    selected_id = sorted(pool, key=lambda model_id: (-float(metric[model_id]["worst_objective_marginal_win_rate"]),
                                                     int(by_id[model_id]["step"]),
                                                     by_id[model_id]["candidate_id"]))[0]
    selected = {**by_id[selected_id], "metric": metric[selected_id],
                "robust_neighbor_selection": bool(robust),
                "full647_gate": next(row for row in gates4096["rows"] if row["model_id"] == selected_id)}
    lock = {
        "status": "OS_SELECTION_LOCKED_BEFORE_FRESH_TEST_EXECUTION",
        "locked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selected": selected, "selection_pool_model_ids": pool,
        "all_4096_gate_eligible_model_ids": eligible_ids,
        "all_2048_gate_eligible_model_ids": eligible2048,
        "gate4096_sha256": sha256(args.gate_summary),
        "gate2048_sha256": sha256(gate_dir / "summary.json"),
        "metric_lock_sha256": sha256(args.new_metric_lock),
        "prior_evaluator_lock_sha256": sha256(args.metric_lock),
        "model_summary_sha256": sha256(results / "model_summary.json"),
        "fixed647_used_for_selection": True, "fresh_test_consulted": False,
        "spent_sealed_split_touched": False,
    }
    atomic_json(args.work / "selection_lock.json", lock)
    atomic_json(args.work / "status.json", {"status": "completed", "selected_model_id": selected_id,
                "selection_lock": str(args.work / "selection_lock.json"),
                "spent_sealed_split_touched": False})
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
