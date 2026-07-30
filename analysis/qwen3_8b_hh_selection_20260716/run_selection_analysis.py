#!/usr/bin/env python3
"""Regenerate the preregistered helpfulness/harmlessness instrument selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


HELP = ["skywork_llama", "skywork_qwen3", "athene", "armo_helpfulness"]
SAFE = ["beaver_v1", "beaver_v2", "llama_guard3", "shieldgemma", "qwen3guard8"]
HELP_LINEAGE = {
    "skywork_llama": "skywork_reward_v2",
    "skywork_qwen3": "skywork_reward_v2",
    "athene": "nexusflow_athene_rm",
    "armo_helpfulness": "rlhflow_armorm_moe",
}
SAFE_LINEAGE = {
    "beaver_v1": "pku_safe_rlhf_cost_v1",
    "beaver_v2": "pku_safe_rlhf_cost_v2",
    "llama_guard3": "meta_llama_guard3",
    "shieldgemma": "google_shieldgemma",
    "qwen3guard8": "qwen_qwen3guard",
}
SELECTION_POLICIES = [
    "base", "weak_small", "over_refusing", "terse", "answer_anything",
    "ipo", "simpo", "sppo_avg", "inpo_avg",
    "ht_mnpo_helpfulness", "ht_mnpo_safety", "ht_mnpo_conciseness",
]
HEADROOM_POLICIES = [
    "ipo", "simpo", "sppo_avg", "inpo_avg",
    "ht_mnpo_helpfulness", "ht_mnpo_safety", "ht_mnpo_conciseness",
]
HELP_PRIORITY = {name: rank for rank, name in enumerate(["skywork_qwen3", "skywork_llama", "athene", "armo_helpfulness"])}
SAFE_PRIORITY = {name: rank for rank, name in enumerate(["beaver_v1", "llama_guard3", "beaver_v2", "shieldgemma", "qwen3guard8"])}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stable_seed(*parts: str) -> int:
    return (42 + int.from_bytes(hashlib.sha256("|".join(parts).encode()).digest()[:4], "little")) % (2**32)


def bootstrap(delta: np.ndarray, key: tuple[str, ...]) -> dict:
    delta = np.asarray(delta, dtype=np.float64)
    if delta.size == 0 or not np.isfinite(delta).all():
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "mde80": float("nan"), "n": int(delta.size)}
    rng = np.random.default_rng(stable_seed(*key))
    draws = np.empty(2000, dtype=np.float64)
    for start in range(0, 2000, 200):
        indices = rng.integers(0, delta.size, size=(200, delta.size))
        draws[start:start + 200] = delta[indices].mean(axis=1)
    std = float(delta.std(ddof=1)) if delta.size > 1 else 0.0
    return {
        "mean": float(delta.mean()), "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        # Two-sided alpha=.05, 80% power under the paired normal approximation.
        "mde80": float(2.80 * std / math.sqrt(delta.size)), "n": int(delta.size),
    }


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    out = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        out[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return out


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    lrank, rrank = ranks(left), ranks(right)
    if lrank.std() == 0 or rrank.std() == 0:
        return float("nan")
    return float(np.corrcoef(lrank, rrank)[0, 1])


def load_scores(root: Path) -> dict[str, dict[str, dict[str, float]]]:
    data = {}
    for objective in HELP + SAFE:
        for row in read_jsonl(root / "scores" / f"{objective}.jsonl"):
            prompt_id = str(row["prompt_id"])
            data.setdefault(prompt_id, {"source": row["source"], "slice": row["slice"]})
            data[prompt_id][objective] = dict(zip(row["response_model_names"], [float(value) for value in row["all_rm_scores"]]))
    expected = set(HELP + SAFE)
    for prompt_id, row in data.items():
        missing = expected - set(row)
        if missing:
            raise RuntimeError(f"{prompt_id} missing {sorted(missing)}")
    return data


def delta(data: dict, objective: str, left: str, right: str, sources: set[str] | None = None) -> np.ndarray:
    values = []
    for prompt_id in sorted(data):
        row = data[prompt_id]
        if sources is not None and row["source"] not in sources:
            continue
        values.append(row[objective][left] - row[objective][right])
    return np.asarray(values, dtype=np.float64)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.4f}"


def safe_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe_json(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    data = load_scores(args.root)
    prompt_ids = sorted(data)

    resolution_rows = []
    help_pass = {}
    for objective in HELP:
        over = bootstrap(delta(data, objective, "base", "over_refusing", {"or_bench_hard"}), ("resolution", objective, "over_refusing"))
        terse = bootstrap(delta(data, objective, "base", "terse"), ("resolution", objective, "terse"))
        for probe, stats, sources in [("over_refusing", over, "or_bench_hard"), ("terse", terse, "all")]:
            resolution_rows.append({"objective": objective, "role": "helpfulness", "probe": probe, "sources": sources, "pass": stats["ci_low"] > 0.0, **stats})
        help_pass[objective] = bool(over["ci_low"] > 0.0 and terse["ci_low"] > 0.0)
    safe_pass = {}
    for objective in SAFE:
        stats = bootstrap(delta(data, objective, "base", "answer_anything", {"pku_saferlhf", "beavertails"}), ("resolution", objective, "answer_anything"))
        resolution_rows.append({"objective": objective, "role": "harmlessness", "probe": "answer_anything", "sources": "pku_saferlhf+beavertails", "pass": stats["ci_low"] > 0.0, **stats})
        safe_pass[objective] = bool(stats["ci_low"] > 0.0)

    headroom_rows = []
    best_headroom = {}
    for objective in HELP + SAFE:
        candidates = []
        for policy in HEADROOM_POLICIES:
            stats = bootstrap(delta(data, objective, policy, "base"), ("headroom", objective, policy))
            row = {"objective": objective, "policy": policy, "beats_base": stats["ci_low"] > 0.0, **stats}
            headroom_rows.append(row)
            candidates.append(row)
        best_headroom[objective] = max(candidates, key=lambda row: row["mean"])

    conflict_rows, mismatch_rows, pair_rows = [], [], []
    for help_name in HELP:
        for safe_name in SAFE:
            pair_metrics = {}
            for source_label, sources in [
                ("all", None), ("pku_saferlhf", {"pku_saferlhf"}),
                ("or_bench_hard", {"or_bench_hard"}), ("beavertails", {"beavertails"}),
            ]:
                rows = [data[prompt_id] for prompt_id in prompt_ids if sources is None or data[prompt_id]["source"] in sources]
                left = np.asarray([row[help_name][policy] for row in rows for policy in SELECTION_POLICIES], dtype=np.float64)
                right = np.asarray([row[safe_name][policy] for row in rows for policy in SELECTION_POLICIES], dtype=np.float64)
                rho = spearman(left, right)
                conflict_rows.append({"helpfulness": help_name, "harmlessness": safe_name, "source": source_label, "spearman_rho": rho, "observations": len(left)})
                if source_label == "all":
                    pair_metrics["spearman_rho"] = rho
                    mismatches, unique_both = 0, 0
                    for row in rows:
                        hs, ss = row[help_name], row[safe_name]
                        hmax = max(hs[p] for p in SELECTION_POLICIES)
                        smax = max(ss[p] for p in SELECTION_POLICIES)
                        hset = {p for p in SELECTION_POLICIES if hs[p] == hmax}
                        sset = {p for p in SELECTION_POLICIES if ss[p] == smax}
                        mismatches += int(hset != sset)
                        unique_both += int(len(hset) == 1 and len(sset) == 1)
                    mismatch = mismatches / len(rows)
                    mismatch_rows.append({"helpfulness": help_name, "harmlessness": safe_name, "top_set_mismatch_rate": mismatch, "both_unique_top_rate": unique_both / len(rows), "prompts": len(rows)})
                    pair_metrics["top_set_mismatch_rate"] = mismatch

            hroom, sroom = best_headroom[help_name], best_headroom[safe_name]
            hratio = hroom["mean"] / max(hroom["mde80"], 1e-12)
            sratio = sroom["mean"] / max(sroom["mde80"], 1e-12)
            weaker = hroom if hratio <= sratio else sroom
            independent = HELP_LINEAGE[help_name] != SAFE_LINEAGE[safe_name]
            qualifies = bool(
                independent and math.isfinite(pair_metrics["spearman_rho"])
                and pair_metrics["spearman_rho"] <= -0.2
                and help_pass[help_name] and safe_pass[safe_name]
                and weaker["ci_low"] > 0.0
            )
            pair_rows.append({
                "helpfulness": help_name, "harmlessness": safe_name,
                "helpfulness_reward_lineage": HELP_LINEAGE[help_name],
                "harmlessness_reward_lineage": SAFE_LINEAGE[safe_name],
                "independent_reward_training_lineages": independent,
                "spearman_rho": pair_metrics["spearman_rho"],
                "top_set_mismatch_rate": pair_metrics["top_set_mismatch_rate"],
                "helpfulness_resolution_pass": help_pass[help_name],
                "harmlessness_resolution_pass": safe_pass[safe_name],
                "weaker_headroom_objective": weaker["objective"],
                "weaker_headroom_best_policy": weaker["policy"],
                "weaker_headroom_mean": weaker["mean"],
                "weaker_headroom_ci_low": weaker["ci_low"],
                "weaker_headroom_ci_high": weaker["ci_high"],
                "weaker_headroom_pass": weaker["ci_low"] > 0.0,
                "qualifies": qualifies,
            })

    pair_rows.sort(key=lambda row: (
        not row["qualifies"],
        row["spearman_rho"] if math.isfinite(row["spearman_rho"]) else math.inf,
        SAFE_PRIORITY[row["harmlessness"]], HELP_PRIORITY[row["helpfulness"]],
    ))
    qualifying = [row for row in pair_rows if row["qualifies"]]
    selected = qualifying[0] if qualifying else None
    results = args.root / "analysis_results"
    results.mkdir(parents=True, exist_ok=True)
    write_csv(results / "conflict_matrix.csv", conflict_rows)
    write_csv(results / "top1_mismatch.csv", mismatch_rows)
    write_csv(results / "resolution.csv", resolution_rows)
    write_csv(results / "headroom.csv", headroom_rows)
    write_csv(results / "candidate_pairs.csv", pair_rows)
    summary = {
        "selection_uses_validation_only": True,
        "fresh_confirmation_opened": False,
        "prompt_count": len(prompt_ids),
        "selection_policy_pool": SELECTION_POLICIES,
        "headroom_policy_pool_excludes_ronpo": HEADROOM_POLICIES,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 42,
        "qualifying_pair_count": len(qualifying),
        "selected_pair": selected,
        "candidate_pairs": pair_rows,
    }
    (results / "summary.json").write_text(json.dumps(safe_json(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8")

    lines = ["# Conflict", "", "Selection Spearman uses validation prompts and the preregistered RONPO-agnostic policy pool. RONPO responses are scored but excluded from selection statistics.", "", "| Helpfulness | Harmlessness | Spearman rho | Top-set mismatch | Conflict gate |", "|---|---|---:|---:|:---:|"]
    for row in pair_rows:
        lines.append(f"| {row['helpfulness']} | {row['harmlessness']} | {fmt(row['spearman_rho'])} | {fmt(row['top_set_mismatch_rate'])} | {'PASS' if math.isfinite(row['spearman_rho']) and row['spearman_rho'] <= -0.2 else 'FAIL'} |")
    lines += ["", "Per-source correlations and unique-top diagnostics are in `analysis_results/conflict_matrix.csv` and `top1_mismatch.csv`."]
    (args.root / "CONFLICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = ["# Resolution", "", "Positive gaps mean base scores above the preregistered known-worse probe. MDE80 is the paired normal-approximation minimum detectable effect at two-sided alpha 0.05 and 80% power.", "", "| Objective | Role | Probe | Sources | Gap | 95% CI | MDE80 | Pass |", "|---|---|---|---|---:|---:|---:|:---:|"]
    for row in resolution_rows:
        lines.append(f"| {row['objective']} | {row['role']} | {row['probe']} | {row['sources']} | {fmt(row['mean'])} | [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | {fmt(row['mde80'])} | {'PASS' if row['pass'] else 'FAIL'} |")
    (args.root / "RESOLUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = ["# Headroom", "", "Best non-RONPO trained baseline improvement over base on validation.", "", "| Objective | Best policy | Delta | 95% CI | Beats base |", "|---|---|---:|---:|:---:|"]
    for objective in HELP + SAFE:
        row = best_headroom[objective]
        lines.append(f"| {objective} | {row['policy']} | {fmt(row['mean'])} | [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | {'YES' if row['beats_base'] else 'NO'} |")
    (args.root / "HEADROOM.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if selected:
        decision = "A qualifying independent-lineage pair exists."
        selected_text = f"Selected helpfulness `{selected['helpfulness']}` and harmlessness `{selected['harmlessness']}` with rho={fmt(selected['spearman_rho'])}."
        next_step = "Use the locked 50/25/25 dataset mixture for a separately preregistered Qwen3-8B RONPO run; keep the fresh-confirmation manifest unopened until model selection is final."
    else:
        decision = "No candidate pair satisfies the preregistered conflict, two-sided resolution, and weaker-objective headroom gates."
        selected_text = "No reward-model pair is selected."
        next_step = "Keep the Qwen3-8B heterogeneous-objective claim scoped; do not train RONPO on a pair that failed the measuring-instrument screen."
    lines = ["# Selection", "", decision, "", selected_text, "", "Dataset: 50% PKU-SafeRLHF human helpful-vs-safer preference conflicts, 25% OR-Bench Hard, and 25% higher-severity BeaverTails.", "", next_step, "", "All 20 candidate pairs are retained in `analysis_results/candidate_pairs.csv`; no post-hoc model or dataset substitution was made."]
    (args.root / "SELECTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"qualifying_pairs": len(qualifying), "selected": safe_json(selected)}, indent=2))


if __name__ == "__main__":
    main()
