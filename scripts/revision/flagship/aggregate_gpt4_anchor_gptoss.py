#!/usr/bin/env python3
"""Aggregate the frozen GPT-4-anchor gpt-oss proxy with prompt bootstrap CIs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPECTED = {"alpaca_eval_2": 805, "arena_hard_v0.1": 500, "mt_bench": 80}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def bootstrap(values: np.ndarray, seed: int = 42, resamples: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=float)
    for index in range(resamples):
        samples[index] = values[rng.integers(0, len(values), size=len(values))].mean()
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def arena_outcome(label: str, candidate_label: str) -> tuple[float, int]:
    if label == "A=B":
        return 0.5, 1
    winner = "A" if label.startswith("A>") else "B"
    significant = ">>" in label
    return (1.0 if winner == candidate_label else 0.0), (3 if significant else 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgment-dir", type=Path, required=True)
    parser.add_argument("--arena-judgment-dir", type=Path)
    parser.add_argument("--arena-protocol-lock", type=Path)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--responses-root", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--alpaca-eval-src", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for path in sorted(args.judgment_dir.glob("shard_*.jsonl")):
        rows.extend(load_jsonl(path))
    arena_protocol = None
    if args.arena_judgment_dir:
        rows = [row for row in rows if row["benchmark"] != "arena_hard_v0.1"]
        for path in sorted(args.arena_judgment_dir.glob("shard_*.jsonl")):
            rows.extend(load_jsonl(path))
        if not args.arena_protocol_lock:
            raise RuntimeError("--arena-protocol-lock is required with --arena-judgment-dir")
        arena_protocol = json.loads(args.arena_protocol_lock.read_text(encoding="utf-8"))
        if arena_protocol["parent_protocol_sha256"] != protocol["configuration_sha256"]:
            raise RuntimeError("Arena adaptation parent protocol mismatch")
    if len(rows) != protocol["expected_total_judgments"]:
        raise RuntimeError(f"expected {protocol['expected_total_judgments']} judgments, got {len(rows)}")
    if len({row["task_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate task ids")
    invalid = [row for row in rows if not row.get("valid")]
    if invalid:
        raise RuntimeError(f"fail closed: {len(invalid)} invalid judge outputs")
    for benchmark, count in protocol["expected_judgments"].items():
        actual = sum(row["benchmark"] == benchmark for row in rows)
        if actual != count:
            raise RuntimeError(f"{benchmark}: expected {count}, got {actual}")

    by_key: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_key[(row["model"], row["benchmark"], row["item_id"])].append(row)
    item_rows: list[dict] = []
    for (model, benchmark, item_id), judgments in sorted(by_key.items()):
        if benchmark == "alpaca_eval_2":
            if len(judgments) != 2 or {row["order"] for row in judgments} != {"reference_first", "candidate_first"}:
                raise RuntimeError(f"missing Alpaca position swap: {model}/{item_id}")
            scores = [float(row["parsed"] == row["candidate_label"]) for row in judgments]
            ref_first = next(row for row in judgments if row["order"] == "reference_first")
            item_rows.append({
                "model": model, "benchmark": benchmark, "item_id": item_id,
                "score": float(np.mean(scores)), "reference_first_score": float(ref_first["parsed"] == ref_first["candidate_label"]),
                "position_agreement": scores[0] == scores[1],
            })
        elif benchmark == "arena_hard_v0.1":
            if len(judgments) != 2 or {row["order"] for row in judgments} != {"reference_first", "candidate_first"}:
                raise RuntimeError(f"missing Arena position swap: {model}/{item_id}")
            outcomes = [arena_outcome(str(row["parsed"]), row["candidate_label"]) for row in judgments]
            numerator = sum(score * weight for score, weight in outcomes)
            denominator = sum(weight for _, weight in outcomes)
            item_rows.append({
                "model": model, "benchmark": benchmark, "item_id": item_id,
                "score": numerator / denominator, "severity_weight": denominator,
                "position_agreement": outcomes[0][0] == outcomes[1][0],
            })
        else:
            if len(judgments) != 2 or {int(row["turn"]) for row in judgments} != {1, 2}:
                raise RuntimeError(f"missing MT-Bench turn: {model}/{item_id}")
            item_rows.append({
                "model": model, "benchmark": benchmark, "item_id": item_id,
                "score": float(np.mean([float(row["parsed"]) for row in judgments])),
                "turn_1_score": float(next(row["parsed"] for row in judgments if int(row["turn"]) == 1)),
                "turn_2_score": float(next(row["parsed"] for row in judgments if int(row["turn"]) == 2)),
            })

    summaries: list[dict] = []
    for model in protocol["models"]:
        for benchmark, expected in EXPECTED.items():
            selected = [row for row in item_rows if row["model"] == model and row["benchmark"] == benchmark]
            if len(selected) != expected:
                raise RuntimeError(f"{model}/{benchmark}: expected {expected} items, got {len(selected)}")
            values = np.asarray([row["score"] for row in selected], dtype=float)
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"nonfinite score: {model}/{benchmark}")
            low, high = bootstrap(values)
            summary = {
                "model": model, "benchmark": benchmark, "n_items": len(values),
                "score": float(values.mean()), "ci95_low": low, "ci95_high": high,
            }
            if benchmark == "alpaca_eval_2":
                summary["reference_first_raw_win_rate"] = float(np.mean([row["reference_first_score"] for row in selected]))
                summary["position_agreement"] = float(np.mean([row["position_agreement"] for row in selected]))
            elif benchmark == "arena_hard_v0.1":
                summary["position_agreement"] = float(np.mean([row["position_agreement"] for row in selected]))
            else:
                summary["turn_1_score"] = float(np.mean([row["turn_1_score"] for row in selected]))
                summary["turn_2_score"] = float(np.mean([row["turn_2_score"] for row in selected]))
            summaries.append(summary)

    lc_status: dict = {"status": "not_requested"}
    if args.alpaca_eval_src:
        try:
            import pandas as pd
            from huggingface_hub import hf_hub_download
            sys.path.insert(0, str(args.alpaca_eval_src / "src"))
            import alpaca_eval.metrics.glm_winrate as glm_module

            pinned_df_gamed = Path(hf_hub_download(
                repo_id=protocol["references"]["alpaca_eval_2"]["repo"],
                repo_type="dataset", filename="df_gamed.csv",
                revision=protocol["references"]["alpaca_eval_2"]["revision"],
            ))
            original_download = glm_module.hf_hub_download
            glm_module.hf_hub_download = lambda *unused_args, **unused_kwargs: str(pinned_df_gamed)
            get_length_controlled_winrate = glm_module.get_length_controlled_winrate

            references = {row["item_id"]: row for row in load_jsonl(args.reference_dir / "alpaca_eval_2_gpt4_reference.jsonl")}
            lc_values = {}
            for model in protocol["models"]:
                responses = {row["item_id"]: row for row in load_jsonl(args.responses_root / model / "responses.jsonl")}
                judged = {
                    row["item_id"]: row for row in rows
                    if row["benchmark"] == "alpaca_eval_2" and row["model"] == model and row["order"] == "reference_first"
                }
                annotations = []
                for item_id, reference in sorted(references.items()):
                    candidate_won = judged[item_id]["parsed"] == judged[item_id]["candidate_label"]
                    annotations.append({
                        "preference": 2 if candidate_won else 1,
                        "output_1": reference["reference_answer"],
                        "output_2": responses[item_id]["responses"][0],
                        "index": int(item_id.rsplit(":", 1)[1]),
                        "generator_1": "gpt4_1106_preview", "generator_2": model,
                        "annotator": "gpt-oss-120b-low-proxy",
                    })
                frame = pd.DataFrame(annotations)
                metrics = get_length_controlled_winrate(
                    frame, save_weights_dir=None, is_add_glm_preference_inplace=False
                )
                lc_values[model] = {
                    "length_controlled_win_rate": float(metrics["length_controlled_winrate"] / 100.0),
                    "lc_standard_error": float(metrics["lc_standard_error"] / 100.0),
                    "reference_first_raw_win_rate": float(metrics["win_rate"] / 100.0),
                }
            for summary in summaries:
                if summary["benchmark"] == "alpaca_eval_2":
                    summary.update(lc_values[summary["model"]])
            lc_status = {
                "status": "complete", "implementation": "official tatsu-lab/alpaca_eval get_length_controlled_winrate",
                "df_gamed_sha256": hashlib.sha256(pinned_df_gamed.read_bytes()).hexdigest(),
                "df_gamed_revision": protocol["references"]["alpaca_eval_2"]["revision"],
                "note": "Judge-replacement proxy, not an official AlpacaEval 2 LC score.",
            }
            glm_module.hf_hub_download = original_download
        except Exception as exc:
            lc_status = {"status": "failed", "error": repr(exc)}

    for benchmark in EXPECTED:
        ranked = sorted((row for row in summaries if row["benchmark"] == benchmark), key=lambda row: row["score"], reverse=True)
        for rank, row in enumerate(ranked, 1):
            row["rank"] = rank
        if benchmark == "alpaca_eval_2" and lc_status["status"] == "complete":
            lc_ranked = sorted(ranked, key=lambda row: row["length_controlled_win_rate"], reverse=True)
            for rank, row in enumerate(lc_ranked, 1):
                row["lc_rank"] = rank

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "item_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({key for row in item_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(item_rows)
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({key for row in summaries for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(summaries)
    wandb_run_id = "gpt4anchor-gptoss120b-aggregate-20260714"
    try:
        import wandb
        run = wandb.init(
            entity="promotion-kim", project="mnpo", id=wandb_run_id, resume="allow",
            name="gpt4anchor-gptoss120b-aggregate", config={"protocol": protocol},
        )
        metrics = {}
        for row in summaries:
            prefix = f"{row['benchmark']}/{row['model']}"
            metrics[f"{prefix}/score"] = row["score"]
            metrics[f"{prefix}/ci95_low"] = row["ci95_low"]
            metrics[f"{prefix}/ci95_high"] = row["ci95_high"]
            metrics[f"{prefix}/rank"] = row["rank"]
            if "length_controlled_win_rate" in row:
                metrics[f"{prefix}/length_controlled_win_rate"] = row["length_controlled_win_rate"]
                metrics[f"{prefix}/lc_rank"] = row["lc_rank"]
        wandb.log(metrics)
        run.summary.update(metrics)
        wandb_url = run.url
        run.finish()
    except Exception as exc:
        raise RuntimeError(f"W&B aggregate logging failed: {exc!r}") from exc

    result = {
        "official_score": False, "disclaimer": protocol["official_score_disclaimer"],
        "judge": protocol["judge"], "references": protocol["references"],
        "num_judgments": len(rows), "bootstrap": protocol["bootstrap"],
        "alpaca_length_control": lc_status, "summaries": summaries,
        "wandb_run_id": wandb_run_id, "wandb_url": wandb_url,
        "arena_adaptation_protocol": arena_protocol,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    by_model = {model: {} for model in protocol["models"]}
    for row in summaries:
        by_model[row["model"]][row["benchmark"]] = row
    lines = [
        "# GPT-4-anchor / gpt-oss-120b open-weight judge proxy", "",
        f"**Not official benchmark scores.** {protocol['official_score_disclaimer']}", "",
        "| Method | Alpaca symmetric WR (95% CI) | Alpaca LC proxy | Arena anchor WR (95% CI) | MT-Bench /10 (95% CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in protocol["models"]:
        a, h, m = by_model[model]["alpaca_eval_2"], by_model[model]["arena_hard_v0.1"], by_model[model]["mt_bench"]
        lc = f"{100*a['length_controlled_win_rate']:.2f} (#{a['lc_rank']})" if "length_controlled_win_rate" in a else "N/A"
        lines.append(
            f"| {model} | {100*a['score']:.2f} [{100*a['ci95_low']:.2f}, {100*a['ci95_high']:.2f}] (#{a['rank']}) | {lc} | "
            f"{100*h['score']:.2f} [{100*h['ci95_low']:.2f}, {100*h['ci95_high']:.2f}] (#{h['rank']}) | "
            f"{m['score']:.3f} [{m['ci95_low']:.3f}, {m['ci95_high']:.3f}] (#{m['rank']}) |"
        )
    lines.extend(["", f"Judgments: {len(rows):,}; prompt-bootstrap: 2,000 resamples, seed 42."])
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
