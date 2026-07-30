#!/usr/bin/env python3
"""Regenerate all resolution, conflict, headroom, and triple-screen numbers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


OBJECTIVES = [
    "skywork_v2", "athene", "armo_whole", "armo_helpfulness",
    "armo_safety", "armo_conciseness", "qwen3guard_safety", "brevity",
]
SPLITS = ["general", "conflict_curated"]
TRAINED = [
    "ronpo_full_expect", "ronpo_k_only", "ipo", "simpo", "sppo_avg", "inpo_avg",
    "ht_mnpo_helpfulness", "ht_mnpo_safety", "ht_mnpo_conciseness",
]
MATCHED = {
    "skywork_v2": ("weak_small", "general"),
    "athene": ("weak_small", "general"),
    "armo_whole": ("weak_small", "general"),
    "armo_helpfulness": ("terse", "general"),
    "armo_safety": ("less_aligned", "conflict_curated"),
    "armo_conciseness": ("verbose", "general"),
    "qwen3guard_safety": ("less_aligned", "conflict_curated"),
    "brevity": ("verbose", "general"),
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stable_seed(*values: str) -> int:
    digest = hashlib.sha256("|".join(values).encode()).digest()
    return (42 + int.from_bytes(digest[:4], "little")) % (2**32)


def bootstrap(delta: np.ndarray, key: tuple[str, ...], resamples: int = 2000) -> dict:
    delta = np.asarray(delta, dtype=np.float64)
    if delta.size == 0 or not np.isfinite(delta).all():
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "mde95": float("nan"), "n": int(delta.size)}
    rng = np.random.default_rng(stable_seed(*key))
    draws = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 200):
        count = min(200, resamples - start)
        indices = rng.integers(0, delta.size, size=(count, delta.size))
        draws[start:start + count] = delta[indices].mean(axis=1)
    std = float(delta.std(ddof=1)) if delta.size > 1 else 0.0
    return {
        "mean": float(delta.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "mde95": float(1.96 * std / math.sqrt(delta.size)),
        "n": int(delta.size),
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = average_ranks(x), average_ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def load_scores(root: Path) -> dict[str, dict[str, dict[str, float]]]:
    data: dict[str, dict[str, dict[str, float]]] = {}
    for split in SPLITS:
        data[split] = {}
        for objective in OBJECTIVES:
            path = root / "scores" / split / f"{objective}.jsonl"
            rows = read_jsonl(path)
            for row in rows:
                prompt = str(row["prompt"])
                names = row["response_model_names"]
                scores = [float(value) for value in row["all_rm_scores"]]
                if objective == "armo_conciseness":
                    scores = [-value for value in scores]
                if len(names) != len(scores):
                    raise ValueError(f"name/score mismatch in {path}")
                data[split].setdefault(prompt, {})[objective] = dict(zip(names, scores))
    return data


def aligned_delta(data: dict, split: str, objective: str, left: str, right: str) -> np.ndarray:
    values = []
    for prompt in sorted(data[split]):
        scores = data[split][prompt][objective]
        values.append(scores[left] - scores[right])
    return np.asarray(values, dtype=np.float64)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.4f}"


def json_safe(value):
    """Represent undefined correlations explicitly without changing any gate."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    data = load_scores(root)

    resolution = []
    for objective in OBJECTIVES:
        probe, primary_split = MATCHED[objective]
        for split in SPLITS:
            stats = bootstrap(aligned_delta(data, split, objective, "base", probe), ("resolution", objective, split))
            resolution.append({
                "objective": objective, "split": split, "probe": probe,
                "is_primary_split": split == primary_split,
                "pass": bool(split == primary_split and stats["ci_low"] > 0.0),
                **stats,
            })
    resolution_primary = {row["objective"]: row for row in resolution if row["is_primary_split"]}

    conflict_rows = []
    mismatch_rows = []
    matrices: dict[str, dict[str, dict[str, float]]] = {}
    for split in SPLITS:
        matrices[split] = {objective: {} for objective in OBJECTIVES}
        prompts = sorted(data[split])
        policy_order = list(next(iter(data[split].values()))[OBJECTIVES[0]])
        pooled = {
            objective: np.asarray([
                data[split][prompt][objective][policy]
                for prompt in prompts for policy in policy_order
            ], dtype=np.float64)
            for objective in OBJECTIVES
        }
        for left in OBJECTIVES:
            for right in OBJECTIVES:
                rho = 1.0 if left == right else spearman(pooled[left], pooled[right])
                matrices[split][left][right] = rho
                conflict_rows.append({"split": split, "objective_left": left, "objective_right": right, "spearman_rho": rho})
        for left, right in itertools.combinations(OBJECTIVES, 2):
            mismatches = 0
            unique_both = 0
            for prompt in prompts:
                ls = data[split][prompt][left]
                rs = data[split][prompt][right]
                lmax, rmax = max(ls.values()), max(rs.values())
                lset = {name for name, value in ls.items() if value == lmax}
                rset = {name for name, value in rs.items() if value == rmax}
                mismatches += int(lset != rset)
                unique_both += int(len(lset) == 1 and len(rset) == 1)
            mismatch_rows.append({
                "split": split, "objective_left": left, "objective_right": right,
                "top_set_mismatch_rate": mismatches / len(prompts),
                "both_unique_top_rate": unique_both / len(prompts),
            })

    headroom = []
    best_headroom: dict[str, dict] = {}
    for objective in OBJECTIVES:
        primary_split = MATCHED[objective][1]
        candidates = []
        for split in SPLITS:
            for policy in TRAINED:
                stats = bootstrap(aligned_delta(data, split, objective, policy, "base"), ("headroom", objective, split, policy))
                row = {
                    "objective": objective, "split": split, "policy": policy,
                    "is_primary_split": split == primary_split,
                    "beats_base": bool(stats["ci_low"] > 0.0), **stats,
                }
                headroom.append(row)
                if split == primary_split:
                    candidates.append(row)
        best_headroom[objective] = max(candidates, key=lambda row: row["mean"])

    triples = []
    for triple in itertools.combinations(OBJECTIVES, 3):
        pair_rhos = [matrices["conflict_curated"][a][b] for a, b in itertools.combinations(triple, 2)]
        resolution_ok = all(resolution_primary[name]["pass"] for name in triple)
        conflict_ok = all(math.isfinite(rho) and rho <= 0.0 for rho in pair_rhos)
        weakest = min((best_headroom[name] for name in triple), key=lambda row: row["mean"])
        headroom_ok = bool(weakest["ci_low"] > 0.0)
        margins = [resolution_primary[name]["mean"] / max(resolution_primary[name]["mde95"], 1e-12) for name in triple]
        triples.append({
            "objectives": list(triple),
            "qualifies": resolution_ok and conflict_ok and headroom_ok,
            "resolution_ok": resolution_ok,
            "conflict_ok": conflict_ok,
            "headroom_ok": headroom_ok,
            "mean_pairwise_conflict_rho": float(np.mean(pair_rhos)) if all(math.isfinite(v) for v in pair_rhos) else float("nan"),
            "max_pairwise_conflict_rho": max(pair_rhos) if all(math.isfinite(v) for v in pair_rhos) else float("nan"),
            "minimum_resolution_margin_over_mde": min(margins),
            "weakest_headroom_objective": weakest["objective"],
            "weakest_headroom_best_policy": weakest["policy"],
            "weakest_headroom_mean": weakest["mean"],
            "weakest_headroom_ci_low": weakest["ci_low"],
            "weakest_headroom_ci_high": weakest["ci_high"],
        })
    triples.sort(key=lambda row: (not row["qualifies"], row["mean_pairwise_conflict_rho"] if math.isfinite(row["mean_pairwise_conflict_rho"]) else math.inf, -row["minimum_resolution_margin_over_mde"]))
    qualifying = [row for row in triples if row["qualifies"]]

    results = root / "analysis_results"
    results.mkdir(parents=True, exist_ok=True)
    write_csv(results / "resolution.csv", resolution)
    write_csv(results / "conflict_matrix.csv", conflict_rows)
    write_csv(results / "top1_mismatch.csv", mismatch_rows)
    write_csv(results / "headroom.csv", headroom)
    write_csv(results / "triples.csv", triples)
    summary = {
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 42,
        "objectives": OBJECTIVES,
        "trained_policies": TRAINED,
        "qualifying_triple_count": len(qualifying),
        "selected_triple": qualifying[0] if qualifying else None,
        "all_triples": triples,
    }
    (results / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8")

    res_lines = ["# Resolution", "", "Positive gaps mean base scores above the pre-registered known-worse probe. PASS is evaluated only on the objective's pre-registered primary diagnostic split.", "", "| Objective | Primary split | Probe | Gap | 95% CI | MDE95 | Pass |", "|---|---|---|---:|---:|---:|:---:|"]
    for objective in OBJECTIVES:
        row = resolution_primary[objective]
        res_lines.append(f"| {objective} | {row['split']} | {row['probe']} | {f(row['mean'])} | [{f(row['ci_low'])}, {f(row['ci_high'])}] | {f(row['mde95'])} | {'PASS' if row['pass'] else 'FAIL'} |")
    (root / "RESOLUTION.md").write_text("\n".join(res_lines) + "\n", encoding="utf-8")

    con_lines = ["# Conflict", "", "Spearman correlations use pooled (prompt, policy) raw scores. Qualification uses the conflict-curated matrix; the general matrix is reported as an external-validity diagnostic."]
    for split in SPLITS:
        con_lines += ["", f"## {split}", "", "| Objective | " + " | ".join(OBJECTIVES) + " |", "|---|" + "---:|" * len(OBJECTIVES)]
        for left in OBJECTIVES:
            con_lines.append("| " + left + " | " + " | ".join(f(matrices[split][left][right]) for right in OBJECTIVES) + " |")
    con_lines += ["", "Pairwise top-set mismatch rates and unique-top rates are in `analysis_results/top1_mismatch.csv`."]
    (root / "CONFLICT.md").write_text("\n".join(con_lines) + "\n", encoding="utf-8")

    head_lines = ["# Headroom", "", "For each objective, this table gives the existing stable trained policy with the largest paired improvement over base on the pre-registered primary diagnostic split.", "", "| Objective | Split | Best policy | Delta | 95% CI | Beats base |", "|---|---|---|---:|---:|:---:|"]
    for objective in OBJECTIVES:
        row = best_headroom[objective]
        head_lines.append(f"| {objective} | {row['split']} | {row['policy']} | {f(row['mean'])} | [{f(row['ci_low'])}, {f(row['ci_high'])}] | {'YES' if row['beats_base'] else 'NO'} |")
    (root / "HEADROOM.md").write_text("\n".join(head_lines) + "\n", encoding="utf-8")

    if qualifying:
        best = qualifying[0]
        decision = "Outcome A: a qualifying objective triple exists."
        recommendation = "Use this locked triple in a subsequent, separately preregistered Qwen3-8B RONPO training study."
    else:
        best = None
        decision = "Outcome B: no objective triple satisfies the pre-registered resolution, conflict, and weakest-objective headroom gates."
        recommendation = "Keep the model-scale robustness claim scoped to 1.5B and synthetic games unless new objective instruments are validated in a new screen."
    dec_lines = ["# Decision", "", decision, "", f"Qualifying triples: {len(qualifying)} of {len(triples)}.", ""]
    if best:
        dec_lines += [f"Selected triple: `{', '.join(best['objectives'])}`.", "", f"Mean pairwise conflict rho: {f(best['mean_pairwise_conflict_rho'])}. Weakest-objective headroom: {best['weakest_headroom_objective']} via {best['weakest_headroom_best_policy']} = {f(best['weakest_headroom_mean'])} [{f(best['weakest_headroom_ci_low'])}, {f(best['weakest_headroom_ci_high'])}].", ""]
    dec_lines += [recommendation, "", "All objectives and all enumerated triples remain available in `analysis_results/`; no objective was hidden after measurement."]
    (root / "DECISION.md").write_text("\n".join(dec_lines) + "\n", encoding="utf-8")
    print(json.dumps({"qualifying_triples": len(qualifying), "selected": best}, indent=2))


if __name__ == "__main__":
    main()
