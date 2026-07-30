#!/usr/bin/env python3
"""Evaluate selected OS with the exact frozen Table-4 models on the common 647 protocol."""

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


PRIOR_SCORE_IDS = {
    "base": "base",
    "ipo": "frozen_ipo",
    "ht_mnpo_conciseness": "frozen_ht_mnpo_conciseness",
}


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
        str(row.get("prompt", "")).strip() and str(row.get("generated_text", "")).strip() for row in rows)


def run(command: list[str], cwd: Path, log: Path, env: dict[str, str] | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--selection-work", type=Path, required=True)
    parser.add_argument("--fixed647", type=Path, required=True)
    parser.add_argument("--prior-fixed647", type=Path, required=True)
    parser.add_argument("--metric-lock", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--general-rm-cache", type=Path, required=True)
    parser.add_argument("--armo-cache", type=Path, required=True)
    args = parser.parse_args()
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    selection = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    if ledger.get("status") != "LOCKED_EXACT_TABLE4_BASELINES_NO_RETRAINING":
        raise RuntimeError("baseline ledger is not frozen")
    if selection.get("status") != "OS_SELECTION_LOCKED_BEFORE_FRESH_TEST_EXECUTION":
        raise RuntimeError("OS checkpoint selection is not locked")
    selected = selection["selected"]
    os_name = "ronpo_os"
    generations = args.work / "generations"; logs = args.work / "logs"
    generations.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    models = list(ledger["rows"])
    for row in models:
        output_dir = generations / row["name"]; output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "output_42.json"
        if row.get("reuse_generation") and not complete(output):
            source = args.project / row["reuse_generation"] if not Path(row["reuse_generation"]).is_absolute() else Path(row["reuse_generation"])
            if not complete(source):
                raise RuntimeError(f"locked reuse generation is invalid: {row['name']}")
            shutil.copy2(source, output)
            metadata = source.parent / "decode_metadata.json"
            if metadata.is_file():
                shutil.copy2(metadata, output_dir / "decode_metadata.json")
    os_source = args.selection_work / "generations_2048" / selected["model_id"] / "output_42.json"
    os_output = generations / os_name / "output_42.json"; os_output.parent.mkdir(parents=True, exist_ok=True)
    if not complete(os_output):
        if not complete(os_source):
            raise RuntimeError("selected OS common-protocol generation is missing")
        shutil.copy2(os_source, os_output)
        meta = os_source.parent / "decode_metadata.json"
        if meta.is_file():
            shutil.copy2(meta, os_output.parent / "decode_metadata.json")
    queued = [row for row in models if not complete(generations / row["name"] / "output_42.json")]
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
            row = queued.pop(0); output_dir = generations / row["name"]
            command = [args.python, "-u", str(args.project / "scripts/revision/flagship/decode_vllm_non_thinking.py"),
                       "--data-dir", str(args.fixed647), "--model", row["snapshot"],
                       "--output-dir", str(output_dir), "--seed", "42", "--temperature", "0.7",
                       "--top-p", "0.9", "--max-new-tokens", "2048", "--max-model-len", "8192",
                       "--gpu-memory-utilization", "0.88"]
            env = env_base.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu
            handle = (logs / f"decode_{row['name']}.log").open("a", encoding="utf-8")
            process = subprocess.Popen(command, cwd=args.project, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT)
            running[gpu] = (row, process, handle, command)
        atomic_json(args.work / "decode_status.json", {"status": "running",
                    "queued": [row["name"] for row in queued],
                    "running": [{"gpu": gpu, "model": value[0]["name"], "pid": value[1].pid}
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
            if rc or not complete(generations / row["name"] / "output_42.json"):
                raise RuntimeError(f"exact baseline decode failed: {row['name']}, rc={rc}")
            del running[gpu]
    base = generations / "base/output_42.json"
    candidates = [row["name"] for row in models if row["name"] != "base"] + [os_name]
    gate_rows = []; gate_dir = args.work / "stability_gates"; gate_dir.mkdir(parents=True, exist_ok=True)
    for model in candidates:
        output = gate_dir / f"{model}.json"
        command = [args.python, str(args.project / "scripts/revision/flagship/stability_gate_corrected.py"),
                   "--base", str(base), "--candidate", str(generations / model / "output_42.json"),
                   "--output", str(output), "--expected-records", "647", "--min-length-ratio", "0.33",
                   "--max-length-ratio", "2.0", "--max-repeat-run", "20"]
        result = subprocess.run(command, cwd=args.project, capture_output=True, text=True)
        (logs / f"gate_{model}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode not in {0, 4} or not output.is_file():
            raise RuntimeError(f"gate execution failed: {model}")
        payload = json.loads(output.read_text(encoding="utf-8"))
        gate_rows.append({"id": model, "method": os_name if model == os_name else next(row["method"] for row in models if row["name"] == model),
                          "passed": payload.get("passed") is True, "status": payload.get("status"),
                          "checks": payload.get("checks"), "candidate": payload.get("candidate")})
    eligible = [row["id"] for row in gate_rows if row["passed"]]
    gates = {"status": "completed_fail_closed", "eligible_candidates": eligible,
             "failed_candidates": [row["id"] for row in gate_rows if not row["passed"]],
             "rows": gate_rows, "spent_sealed_split_touched": False}
    atomic_json(gate_dir / "summary.json", gates)
    merged = args.work / "merged_generations.json"
    merge = [args.python, "-m", "mnpo_scripts.merge_model_generations", "--generations", f"base={base}"]
    merge.extend(f"{model}={generations / model / 'output_42.json'}" for model in eligible)
    merge.extend(["--output_file", str(merged)])
    run(merge, args.project, logs / "merge.log")

    cacheable = {name for name in PRIOR_SCORE_IDS if name in eligible or name == "base"}
    if os_name in eligible:
        cacheable.add(os_name)
    new_models = [model for model in eligible if model not in cacheable]
    if new_models:
        new_merged = args.work / "new_score_generations.json"
        new_merge = [args.python, "-m", "mnpo_scripts.merge_model_generations", "--generations"]
        new_merge.extend(f"{model}={generations / model / 'output_42.json'}" for model in new_models)
        new_merge.extend(["--output_file", str(new_merged)])
        run(new_merge, args.project, logs / "merge_new_scores.log")
        run([args.python, str(args.project / "scripts/revision/flagship/score_stage2_os_localrm.py"),
             "--project", str(args.project), "--python", args.python, "--merged", str(new_merged),
             "--work", str(args.work / "new_score_work"), "--general-rm-cache", str(args.general_rm_cache),
             "--armo-cache", str(args.armo_cache), "--expected-prompts", "647"],
            args.project, logs / "score_new_models.log")
    combine = [args.python, str(args.project / "scripts/revision/flagship/combine_cached_localrm_scores.py"),
               "--merged", str(merged), "--objective", "skywork", "--objective", "athene",
               "--objective", "armo"]
    for objective in ("skywork", "athene", "armo"):
        combine.extend(["--source", f"prior:{objective}={args.prior_fixed647 / 'scores' / (objective + '.jsonl')}"])
        combine.extend(["--source", f"selection:{objective}={args.selection_work / 'scores' / (objective + '.jsonl')}"])
        if new_models:
            combine.extend(["--source", f"new:{objective}={args.work / 'new_score_work/scores' / (objective + '.jsonl')}"])
    combine.extend(["--map", "base=prior:base"])
    for model in eligible:
        if model in PRIOR_SCORE_IDS:
            combine.extend(["--map", f"{model}=prior:{PRIOR_SCORE_IDS[model]}"])
        elif model == os_name:
            combine.extend(["--map", f"{model}=selection:{selected['model_id']}"])
        else:
            combine.extend(["--map", f"{model}=new:{model}"])
    combined_scores = args.work / "scores"
    combine.extend(["--output-dir", str(combined_scores)])
    run(combine, args.project, logs / "combine_cached_scores.log")
    grid_rows = []
    by_name = {row["name"]: row for row in models}
    for model in eligible:
        if model == os_name:
            grid_rows.append({"id": model, "method": "ronpo_os", "stage": 1,
                              "model_path": selected["model_path"], "source": "selected_os_only_repair"})
        else:
            row = by_name[model]
            grid_rows.append({"id": model, "method": row["method"], "stage": row["stage"],
                              "model_path": row["snapshot"], "source": "frozen_public_table4_revision"})
    grid = {"status": "frozen_exact_table4_plus_os", "candidates": grid_rows,
            "spent_sealed_split_touched": False}
    grid_path = args.work / "combined_grid.json"; atomic_json(grid_path, grid)
    results = args.work / "results"
    run([args.python, str(args.project / "scripts/revision/flagship/aggregate_stage2_os_localrm.py"),
         "--merged", str(merged), "--score", f"skywork={combined_scores / 'skywork.jsonl'}",
         "--score", f"athene={combined_scores / 'athene.jsonl'}",
         "--score", f"armo={combined_scores / 'armo.jsonl'}", "--grid", str(grid_path),
         "--gates", str(gate_dir / "summary.json"), "--metric-lock", str(args.metric_lock),
         "--output-dir", str(results), "--split-name", "fixed_647_exact_table4_plus_selected_os",
         "--locked-model-set"], args.project, logs / "aggregate.log")
    atomic_json(args.work / "status.json", {
        "status": "completed", "selected_os_model_id": selected["model_id"],
        "eligible_models": ["base", *eligible], "failed_models": gates["failed_candidates"],
        "cached_score_models": sorted(cacheable), "newly_scored_models": new_models,
        "model_summary": str(results / "model_summary.json"),
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "spent_sealed_split_touched": False})


if __name__ == "__main__":
    main()
