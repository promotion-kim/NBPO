#!/usr/bin/env python3
"""Render measured fail-closed artifacts for the corrected confirmatory run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    args = parser.parse_args()
    work = args.eval_root / "confirmatory"
    status = load(work / "status.json")
    protocol = load(args.eval_root / "corrected_protocol_lock.json")
    opened = load(work / "confirmatory_opened.json")
    gates = load(work / "stability_gates/summary.json")
    if status.get("status") != "failed" or status.get("stage") != "confirmatory_stability_gates":
        raise RuntimeError("expected a terminal confirmatory stability-gate failure")

    rows = []
    for model, gate in gates["models"].items():
        candidate = gate["candidate"]
        rows.append({
            "model": model,
            "records": int(candidate["records"]),
            "empty_count": int(candidate["empty_count"]),
            "think_leak_count": int(candidate["think_leak_count"]),
            "mean_word_ratio_vs_base": float(gate["candidate_base_mean_word_ratio"]),
            "max_repeat_run": int(candidate["max_repeat_run"]),
            "status": gate["status"],
        })
    diagnostics = {
        "status": "failed_closed",
        "stage": "confirmatory_stability_gates",
        "generated_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evaluation_split": "all unused prompts in original non-training validation remainder",
        "prompt_count": int(opened["prompt_count"]),
        "prompt_file_sha256": opened["confirmatory_file_sha256"],
        "opened_at_kst": opened["opened_at_kst"],
        "selected_ronpo_variant": "top-mass",
        "selected_model_name": "ronpo_k_only",
        "model_selection_changed_after_source_test": False,
        "decode": protocol["decode"],
        "stability_rows": rows,
        "failed_models": [row["model"] for row in rows if row["status"] != "passed"],
        "reward_scoring_started": False,
        "normalization_started": False,
        "bootstrap_started": False,
        "confirmatory_rank_measured": False,
        "reason": "Six models exceeded the frozen max-repeat-run threshold; fail-closed stopped before ArmoRM scoring.",
        "provenance_limitations": protocol["provenance_limitations"],
    }
    out = work / "results"
    out.mkdir(exist_ok=True)
    (out / "failure_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")

    lines = [
        "# Corrected confirmatory reward evaluation — terminal fail-closed", "",
        f"The corrected protocol was frozen only after all 11 models passed the 130-prompt "
        f"non-headline validation/stress set. The {diagnostics['prompt_count']}-prompt confirmatory "
        f"holdout was then opened once at `{diagnostics['opened_at_kst']}`.", "",
        "All 11 models generated every prompt with zero empty responses and zero think-tag leakage. "
        "Six models nevertheless exceeded the frozen max-repeat-run threshold. The run stopped before "
        "ArmoRM scoring, per-prompt normalization, bootstrap, or ranking. No confirmatory reward rank exists.", "",
        "| Model | Records | Empty | Think leaks | Mean-word ratio | Max repeat | Gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['records']} | {row['empty_count']} | {row['think_leak_count']} | "
            f"{row['mean_word_ratio_vs_base']:.6f} | {row['max_repeat_run']} | {row['status']} |"
        )
    lines.extend([
        "", "## Frozen protocol", "",
        f"- Prompt-file SHA-256: `{diagnostics['prompt_file_sha256']}`",
        "- Decode: vLLM 0.24.0, seed 42, temperature 0.7, top-p 0.9, max_new_tokens 2048, "
        "enable_thinking=false, bad words `<think>`/`</think>`, repetition pattern sizes 1--4 "
        "with minimum count 20.",
        "- Gate: records 1,736; empty 0; think leaks 0; mean-word ratio [0.33, 2.0]; max-repeat-run <=20.",
        "- Model selection remained the pre-source-test lock: `ronpo_k_only` (top-mass).", "",
        "## Outcome", "",
        "- `ranked_confirmatory_summary.json`: not produced",
        "- `per_objective_scores.csv`: not produced",
        "- Reward scores / normalization / bootstrap: not started",
        "- RONPO confirmatory worst-objective rank: unknown",
        "- Passing S3 models: base, ronpo_full_expect, ronpo_k_only, dpo, ht_mnpo_conciseness",
        "- The holdout will not be regenerated or used to change the protocol.", "",
        "## Limitation", "",
        "This confirmatory holdout is the unused remainder of the original non-training validation partition, "
        "not the already consumed source-test partition.", "",
    ])
    (out / "CONFIRMATORY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
