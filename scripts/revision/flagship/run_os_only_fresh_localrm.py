#!/usr/bin/env python3
"""Open the locked OS-only fresh split once, decode the exact model set, gate, and score local RMs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


def complete(path: Path, expected: int = 128) -> bool:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(rows, list) and len(rows) == expected and all(
        str(row.get("prompt", "")).strip() and str(row.get("generated_text", "")).strip() for row in rows)


def run(command: list[str], cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--fresh-manifest", type=Path, required=True)
    parser.add_argument("--metric-lock", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--general-rm-cache", type=Path, required=True)
    parser.add_argument("--armo-cache", type=Path, required=True)
    args = parser.parse_args()
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    selection = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    fresh = json.loads(args.fresh_manifest.read_text(encoding="utf-8"))
    if ledger.get("status") != "LOCKED_EXACT_TABLE4_BASELINES_NO_RETRAINING":
        raise RuntimeError("baseline ledger is not locked")
    if selection.get("status") != "OS_SELECTION_LOCKED_BEFORE_FRESH_TEST_EXECUTION":
        raise RuntimeError("OS selection is not locked")
    if fresh.get("status") != "FRESH_TEST_PROMPTS_LOCKED_UNOPENED_BEFORE_OS_RANKING" or fresh.get("fresh_test_opened") is not False:
        raise RuntimeError("fresh test is not locked and unopened")
    prompts = Path(fresh["prompt_file"])
    if not prompts.is_absolute():
        prompts = args.project / prompts
    if sha256(prompts) != fresh["prompt_file_sha256"]:
        raise RuntimeError("fresh prompt hash mismatch")
    work = args.run_dir / "fresh_test"; logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    os_row = {"name": "ronpo_os", "method": "ronpo_os", "stage": 1,
              "snapshot": selection["selected"]["model_path"], "training_frozen": True}
    models = [*ledger["rows"], os_row]
    opened = args.run_dir / "fresh_opened.json"
    model_lock = [{"name": row["name"], "method": row["method"], "stage": row["stage"],
                   "model_path": row["snapshot"],
                   "revision": row.get("revision", selection["selected"]["model_id"])} for row in models]
    if opened.is_file():
        value = json.loads(opened.read_text(encoding="utf-8"))
        if value.get("fresh_manifest_sha256") != sha256(args.fresh_manifest) or value.get("models") != model_lock:
            raise RuntimeError("fresh execution lock differs; refusing a second opening")
    else:
        atomic_json(opened, {"status": "FRESH_OPENED_ONCE_FOR_LOCKED_MODEL_SET",
                    "opened_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "fresh_manifest_sha256": sha256(args.fresh_manifest),
                    "prompt_file_sha256": sha256(prompts), "models": model_lock,
                    "decode": {"seed": 42, "temperature": 0.7, "top_p": 0.9,
                               "max_new_tokens": 2048, "dtype": "bfloat16", "enable_thinking": False},
                    "spent_sealed_split_touched": False})
    generations = work / "generations"; generations.mkdir(parents=True, exist_ok=True)
    queued = [row for row in models if not complete(generations / row["name"] / "output_42.json")]
    env_base = os.environ.copy()
    env_base.update({"PYTHONPATH": str(args.project), "HF_HOME": str(args.hf_cache.parent),
                     "HF_HUB_CACHE": str(args.hf_cache), "HF_HUB_OFFLINE": "1",
                     "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                     "TORCH_CUDNN_SDPA_ENABLED": "0", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
    running = {}; failures = []
    while queued or running:
        for gpu in ("0", "1", "2", "3"):
            if not queued or gpu in running:
                continue
            row = queued.pop(0); output = generations / row["name"]; output.mkdir(parents=True, exist_ok=True)
            command = [args.python, "-u", str(args.project / "scripts/revision/flagship/decode_vllm_non_thinking.py"),
                       "--data-dir", str(prompts), "--model", row["snapshot"], "--output-dir", str(output),
                       "--seed", "42", "--temperature", "0.7", "--top-p", "0.9",
                       "--max-new-tokens", "2048", "--max-model-len", "8192",
                       "--gpu-memory-utilization", "0.88"]
            env = env_base.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu
            handle = (logs / f"decode_{row['name']}.log").open("a", encoding="utf-8")
            process = subprocess.Popen(command, cwd=args.project, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT)
            running[gpu] = (row, process, handle, command)
        atomic_json(work / "decode_status.json", {"status": "running",
                    "queued": [row["name"] for row in queued],
                    "running": [{"gpu": gpu, "model": value[0]["name"], "pid": value[1].pid}
                                for gpu, value in running.items()],
                    "failures": failures, "spent_sealed_split_touched": False})
        if not running:
            continue
        time.sleep(5)
        for gpu, (row, process, handle, command) in list(running.items()):
            rc = process.poll()
            if rc is None:
                continue
            handle.close()
            if rc or not complete(generations / row["name"] / "output_42.json"):
                failures.append({"model": row["name"], "returncode": rc, "command": command})
            del running[gpu]
        if failures:
            atomic_json(work / "decode_status.json", {"status": "failed", "failures": failures,
                        "spent_sealed_split_touched": False})
            raise RuntimeError(json.dumps(failures, indent=2))
    base = generations / "base/output_42.json"
    gate_rows = []; gate_dir = work / "stability_gates"; gate_dir.mkdir(parents=True, exist_ok=True)
    for row in models:
        if row["name"] == "base":
            continue
        output = gate_dir / f"{row['name']}.json"
        command = [args.python, str(args.project / "scripts/revision/flagship/stability_gate_corrected.py"),
                   "--base", str(base), "--candidate", str(generations / row["name"] / "output_42.json"),
                   "--output", str(output), "--expected-records", "128", "--min-length-ratio", "0.33",
                   "--max-length-ratio", "2.0", "--max-repeat-run", "20"]
        result = subprocess.run(command, cwd=args.project, capture_output=True, text=True)
        (logs / f"gate_{row['name']}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode not in {0, 4} or not output.is_file():
            raise RuntimeError(f"fresh gate execution failed: {row['name']}")
        payload = json.loads(output.read_text(encoding="utf-8"))
        gate_rows.append({"id": row["name"], "method": row["method"],
                          "passed": payload.get("passed") is True, "status": payload.get("status"),
                          "checks": payload.get("checks"), "candidate": payload.get("candidate")})
    eligible = [row["id"] for row in gate_rows if row["passed"]]
    gates = {"status": "completed_fail_closed", "eligible_models": ["base", *eligible],
             "eligible_candidates": eligible,
             "failed_models": [row["id"] for row in gate_rows if not row["passed"]],
             "failed_candidates": [row["id"] for row in gate_rows if not row["passed"]],
             "rows": gate_rows, "spent_sealed_split_touched": False}
    atomic_json(gate_dir / "summary.json", gates)
    merged = work / "merged_generations.json"
    merge = [args.python, "-m", "mnpo_scripts.merge_model_generations", "--generations", f"base={base}"]
    merge.extend(f"{model}={generations / model / 'output_42.json'}" for model in eligible)
    merge.extend(["--output_file", str(merged)])
    run(merge, args.project, logs / "merge.log")
    run([args.python, str(args.project / "scripts/revision/flagship/score_stage2_os_localrm.py"),
         "--project", str(args.project), "--python", args.python, "--merged", str(merged),
         "--work", str(work / "localrm"), "--general-rm-cache", str(args.general_rm_cache),
         "--armo-cache", str(args.armo_cache), "--expected-prompts", "128"],
        args.project, logs / "score_localrm.log")
    grid = {"status": "frozen_fresh_exact_table4_plus_os", "candidates": [],
            "spent_sealed_split_touched": False}
    for row in models:
        if row["name"] == "base" or row["name"] not in eligible:
            continue
        grid["candidates"].append({"id": row["name"], "method": row["method"], "stage": row["stage"],
                                   "model_path": row["snapshot"],
                                   "source": "selected_os_only_repair" if row["name"] == "ronpo_os" else "frozen_public_table4_revision"})
    grid_path = work / "grid.json"; atomic_json(grid_path, grid)
    results = work / "results/localrm"
    run([args.python, str(args.project / "scripts/revision/flagship/aggregate_stage2_os_localrm.py"),
         "--merged", str(merged), "--score", f"skywork={work / 'localrm/scores/skywork.jsonl'}",
         "--score", f"athene={work / 'localrm/scores/athene.jsonl'}",
         "--score", f"armo={work / 'localrm/scores/armo.jsonl'}", "--grid", str(grid_path),
         "--gates", str(gate_dir / "summary.json"), "--metric-lock", str(args.metric_lock),
         "--output-dir", str(results), "--split-name", "fresh_128_exact_table4_plus_selected_os",
         "--locked-model-set"], args.project, logs / "aggregate_localrm.log")
    summary = json.loads((results / "model_summary.json").read_text(encoding="utf-8"))
    by_model = {row["model"]: row for row in summary["ranked_all_eligible_candidates"]}
    if "ronpo_os" not in by_model:
        decision = "FAIL"
        reason = "selected OS failed the fresh stability gate"
    else:
        os_value = float(by_model["ronpo_os"]["worst_objective_marginal_win_rate"])
        trained = [float(row["worst_objective_marginal_win_rate"]) for row in summary["ranked_all_eligible_candidates"]
                   if row["model"] not in {"base", "ronpo_os"}]
        best_trained = max(trained) if trained else float("-inf")
        if os_value > max(0.5, best_trained):
            decision, reason = "PASS", "OS is strictly highest among every fresh gate-passing model"
        elif os_value > best_trained and os_value <= 0.5:
            decision, reason = "PARTIAL", "OS leads every fresh gate-passing trained method but not base"
        else:
            decision, reason = "FAIL", "a fresh gate-passing baseline ties or exceeds OS"
    atomic_json(args.run_dir / "fresh_localrm_decision.json", {
        "status": "LOCKED_ONE_SHOT_LOCALRM_DECISION", "decision": decision, "reason": reason,
        "selected_os": selection["selected"]["model_id"],
        "model_summary": str(results / "model_summary.json"),
        "model_summary_sha256": sha256(results / "model_summary.json"),
        "fresh_manifest_sha256": sha256(args.fresh_manifest),
        "independent_panel_pending": True,
        "spent_sealed_split_touched": False})
    atomic_json(work / "status.json", {"status": "completed_localrm_pending_panel",
                "eligible_models": ["base", *eligible], "failed_models": gates["failed_models"],
                "decision": decision, "spent_sealed_split_touched": False})
    print(json.dumps({"decision": decision, "reason": reason, "eligible": ["base", *eligible]}, indent=2))


if __name__ == "__main__":
    main()
