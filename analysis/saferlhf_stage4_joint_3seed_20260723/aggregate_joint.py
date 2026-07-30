#!/usr/bin/env python3
"""Aggregate the preregistered SafeRLHF Stage-4 joint three-seed pass."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp


SEEDS = (42, 43, 44)
OBJECTIVES = ("helpfulness", "harmlessness")
BASELINES = (
    "inpo_avg", "sppo_avg", "simpo", "ipo", "dpo",
    "ht_mnpo_harmless", "ht_mnpo_helpfulness",
)
LABELS = {
    "base": "Base",
    "ronpo_os": "RONPO (OS)",
    "ronpo_topmass": "RONPO (top-mass)",
    "inpo_avg": "INPO-avg",
    "sppo_avg": "SPPO-avg",
    "simpo": "SimPO",
    "ipo": "IPO",
    "dpo": "DPO",
    "ht_mnpo_harmless": "HT-MNPO (harml.)",
    "ht_mnpo_helpfulness": "HT-MNPO (help.)",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_score(path: Path, prompt_ids: list[str], names: list[str]) -> np.ndarray:
    rows = read_jsonl(path)
    by_id = {row["prompt_id"]: row for row in rows}
    if len(rows) != len(prompt_ids) or len(by_id) != len(prompt_ids) or set(by_id) != set(prompt_ids):
        raise RuntimeError(f"{path}: prompt mismatch")
    values = []
    for prompt_id in prompt_ids:
        row = by_id[prompt_id]
        if row["response_model_names"] != names:
            raise RuntimeError(f"{path}: policy order mismatch")
        values.append(row["all_rm_scores"])
    array = np.asarray(values, dtype=float)
    if array.shape != (len(prompt_ids), len(names)) or not np.isfinite(array).all():
        raise RuntimeError(f"{path}: invalid score matrix {array.shape}")
    return array


def prompt_ci(values: np.ndarray, rng_seed: int = 42, reps: int = 2000) -> list[float]:
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(values), size=(reps, len(values)))
    return [float(x) for x in np.quantile(values[idx].mean(axis=1), [0.025, 0.975])]


def hierarchical_draws(arrays: list[np.ndarray], reps: int = 10000, rng_seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    out = np.empty(reps)
    for rep in range(reps):
        selected = rng.integers(0, len(arrays), size=len(arrays))
        means = []
        for seed_index in selected:
            values = arrays[int(seed_index)]
            idx = rng.integers(0, len(values), size=len(values))
            means.append(float(values[idx].mean()))
        out[rep] = float(np.mean(means))
    return out


def hierarchical_matrix_draws(arrays: list[np.ndarray], reps: int = 10000, rng_seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    out = np.empty((reps, arrays[0].shape[1]))
    for rep in range(reps):
        selected = rng.integers(0, len(arrays), size=len(arrays))
        means = []
        for seed_index in selected:
            values = arrays[int(seed_index)]
            idx = rng.integers(0, len(values), size=len(values))
            means.append(values[idx].mean(axis=0))
        out[rep] = np.mean(means, axis=0)
    return out


def ci(draws: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def per_method_indices(names: list[str], method: str) -> list[int]:
    return [names.index(f"{method}_s{seed}") for seed in SEEDS]


def scalar_method_summary(matrix: np.ndarray, names: list[str], method: str) -> dict:
    indices = per_method_indices(names, method)
    arrays = [matrix[:, index] for index in indices]
    draws = hierarchical_draws(arrays)
    per_seed = {}
    for seed, values in zip(SEEDS, arrays):
        per_seed[str(seed)] = {"mean": float(values.mean()), "ci95": prompt_ci(values)}
    return {
        "mean": float(np.mean([values.mean() for values in arrays])),
        "hierarchical_ci95": ci(draws),
        "per_seed": per_seed,
    }


def contrast_summary(matrix: np.ndarray, names: list[str], left: str, right: str) -> dict:
    left_indices = per_method_indices(names, left)
    right_indices = per_method_indices(names, right)
    arrays = [matrix[:, li] - matrix[:, ri] for li, ri in zip(left_indices, right_indices)]
    draws = hierarchical_draws(arrays)
    per_seed = {}
    for seed, values in zip(SEEDS, arrays):
        per_seed[str(seed)] = {"delta": float(values.mean()), "ci95": prompt_ci(values)}
    return {
        "delta": float(np.mean([values.mean() for values in arrays])),
        "hierarchical_ci95": ci(draws),
        "per_seed": per_seed,
    }


def objective_method_summary(
    scores: np.ndarray,
    names: list[str],
    method: str,
    base_index: int,
    objective_names: tuple[str, str] = OBJECTIVES,
) -> dict:
    indices = per_method_indices(names, method)
    raw_arrays = [scores[:, :, index] for index in indices]
    win_arrays = [
        np.where(values > scores[:, :, base_index], 1.0,
                 np.where(values < scores[:, :, base_index], 0.0, 0.5))
        for values in raw_arrays
    ]
    raw_draws = hierarchical_matrix_draws(raw_arrays)
    win_draws = hierarchical_matrix_draws(win_arrays)
    objective_means = np.mean([values.mean(axis=0) for values in raw_arrays], axis=0)
    win_means = np.mean([values.mean(axis=0) for values in win_arrays], axis=0)
    return {
        "raw_objectives": dict(zip(objective_names, map(float, objective_means))),
        "raw_objective_hierarchical_ci95": {
            objective: ci(raw_draws[:, index]) for index, objective in enumerate(objective_names)
        },
        "marginal_win_rates_vs_base": dict(zip(objective_names, map(float, win_means))),
        "marginal_win_rate_hierarchical_ci95": {
            objective: ci(win_draws[:, index]) for index, objective in enumerate(objective_names)
        },
        "worst_marginal_win_rate": float(win_means.min()),
        "worst_marginal_win_rate_hierarchical_ci95": ci(win_draws.min(axis=1)),
        "per_seed": {
            str(seed): {
                "raw_objectives": dict(zip(objective_names, map(float, raw.mean(axis=0)))),
                "marginal_win_rates_vs_base": dict(zip(objective_names, map(float, win.mean(axis=0)))),
                "worst_marginal_win_rate": float(win.mean(axis=0).min()),
            }
            for seed, raw, win in zip(SEEDS, raw_arrays, win_arrays)
        },
    }


def decorate(values: dict[str, float]) -> dict[str, str]:
    order = sorted(values, key=values.get, reverse=True)
    out = {name: f"{value:.4f}" for name, value in values.items()}
    if order:
        out[order[0]] = f"\\textbf{{{out[order[0]]}}}"
    if len(order) > 1:
        out[order[1]] = f"\\underline{{{out[order[1]]}}}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--beaver-help", type=Path, required=True)
    parser.add_argument("--beaver-safe", type=Path, required=True)
    parser.add_argument("--independent-quality", type=Path, required=True)
    parser.add_argument("--independent-safety", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    pool = read_jsonl(args.pool)
    names = lock["policy_order"]
    if sha256(args.pool) != lock["response_pool_sha256"]:
        raise RuntimeError("pool hash differs from preregistration")
    prompt_ids = [row["prompt_id"] for row in pool]
    score_paths = {
        "beaver_helpfulness": args.beaver_help,
        "beaver_harmlessness": args.beaver_safe,
        "independent_quality": args.independent_quality,
        "independent_safety": args.independent_safety,
    }
    matrices = {name: load_score(path, prompt_ids, names) for name, path in score_paths.items()}
    beaver = np.stack([matrices["beaver_helpfulness"], matrices["beaver_harmlessness"]], axis=1)
    independent = np.stack([matrices["independent_quality"], matrices["independent_safety"]], axis=1)
    base_index = names.index("base")

    lo = beaver.min(axis=2, keepdims=True)
    hi = beaver.max(axis=2, keepdims=True)
    norm = np.divide(beaver - lo, hi - lo, out=np.full_like(beaver, 0.5), where=(hi - lo) > 0)
    legacy_worst = norm.min(axis=1)
    legacy_avg = norm.mean(axis=1)
    costs = expit(float(lock["primary"]["preference_scale"]) * (
        norm[:, :, :, None] - norm[:, :, None, :]
    ))
    hard = costs.min(axis=(1, 3))
    kappa = float(lock["soft_floor_kappa"])
    soft = -kappa * (
        logsumexp(-costs / kappa, axis=(1, 3)) - np.log(costs.shape[1] * costs.shape[3])
    )

    methods = lock["methods"]
    per_policy = {}
    for index, name in enumerate(names):
        per_policy[name] = {
            "hard_pairwise_floor": float(hard[:, index].mean()),
            "hard_pairwise_floor_ci95": prompt_ci(hard[:, index]),
            "soft_pairwise_floor": float(soft[:, index].mean()),
            "legacy_worst_norm": float(legacy_worst[:, index].mean()),
            "legacy_average_norm": float(legacy_avg[:, index].mean()),
            "beaver_raw": dict(zip(OBJECTIVES, map(float, beaver[:, :, index].mean(axis=0)))),
            "independent_raw": dict(zip(("quality", "safety"), map(float, independent[:, :, index].mean(axis=0)))),
        }

    beaver_methods = {}
    independent_methods = {}
    for method in methods:
        beaver_methods[method] = {
            "hard_pairwise_floor": scalar_method_summary(hard, names, method),
            "soft_pairwise_floor": scalar_method_summary(soft, names, method),
            "legacy_worst_norm": scalar_method_summary(legacy_worst, names, method),
            "legacy_average_norm": scalar_method_summary(legacy_avg, names, method),
            **objective_method_summary(beaver, names, method, base_index),
        }
        independent_methods[method] = objective_method_summary(
            independent, names, method, base_index, ("quality", "safety")
        )

    all_contrasts = {}
    for right in ("ronpo_topmass", *BASELINES):
        all_contrasts[f"ronpo_os_minus_{right}"] = {
            "hard_pairwise_floor": contrast_summary(hard, names, "ronpo_os", right),
            "soft_pairwise_floor": contrast_summary(soft, names, "ronpo_os", right),
            "legacy_worst_norm": contrast_summary(legacy_worst, names, "ronpo_os", right),
            "legacy_average_norm": contrast_summary(legacy_avg, names, "ronpo_os", right),
        }
    primary = all_contrasts["ronpo_os_minus_ipo"]["hard_pairwise_floor"]
    primary_pass = (
        primary["hierarchical_ci95"][0] > 0
        and all(item["delta"] > 0 for item in primary["per_seed"].values())
    )
    independent_ronpo = independent_methods["ronpo_os"]
    independent_pass = (
        all(value > 0.5 for value in independent_ronpo["marginal_win_rates_vs_base"].values())
        and independent_ronpo["worst_marginal_win_rate_hierarchical_ci95"][0] > 0.5
    )

    args.output.mkdir(parents=True, exist_ok=True)
    paired = {
        "status": "complete",
        "records": len(pool),
        "policy_order": names,
        "normalization": "per-prompt per-objective minmax over frozen 28-policy pool",
        "per_policy": per_policy,
        "method_summary": beaver_methods,
        "contrasts": all_contrasts,
        "primary_contrast": "ronpo_os_minus_ipo",
        "primary_pass": primary_pass,
        "bootstrap": lock["bootstrap"],
        "spent_sealed_split_touched": False,
    }
    floor = {
        "formula": "mean_x min_k,a sigmoid(8*(normalized_r_k(y)-normalized_r_k(a)))",
        "opponent_pool": "the frozen 28-policy response set",
        "soft_floor_kappa": kappa,
        "method_summary": {
            method: {
                "hard_pairwise_floor": values["hard_pairwise_floor"],
                "soft_pairwise_floor": values["soft_pairwise_floor"],
            }
            for method, values in beaver_methods.items()
        },
        "contrasts": all_contrasts,
        "primary_pass": primary_pass,
    }
    independent_out = {
        "evaluators": {
            "quality": lock["evaluators"]["independent_quality"],
            "safety": lock["evaluators"]["independent_safety"],
        },
        "metric": "minimum per-objective marginal win rate versus Base",
        "method_summary": independent_methods,
        "ronpo_non_regression_support": independent_pass,
        "spent_sealed_split_touched": False,
    }
    (args.output / "paired_summary.json").write_text(json.dumps(paired, indent=2) + "\n", encoding="utf-8")
    (args.output / "floor_summary.json").write_text(json.dumps(floor, indent=2) + "\n", encoding="utf-8")
    (args.output / "independent_summary.json").write_text(json.dumps(independent_out, indent=2) + "\n", encoding="utf-8")

    with (args.output / "per_policy_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "policy", "beaver_helpfulness_raw", "beaver_harmlessness_raw",
            "legacy_avg_norm", "legacy_worst_norm", "hard_pairwise_floor",
            "soft_pairwise_floor", "independent_quality_raw", "independent_safety_raw",
        ])
        for name in names:
            row = per_policy[name]
            writer.writerow([
                name, row["beaver_raw"]["helpfulness"], row["beaver_raw"]["harmlessness"],
                row["legacy_average_norm"], row["legacy_worst_norm"], row["hard_pairwise_floor"],
                row["soft_pairwise_floor"], row["independent_raw"]["quality"],
                row["independent_raw"]["safety"],
            ])

    display_methods = ["ronpo_os", "ipo", "dpo", "simpo", "inpo_avg", "sppo_avg",
                       "ht_mnpo_helpfulness", "ht_mnpo_harmless"]
    hard_values = {method: beaver_methods[method]["hard_pairwise_floor"]["mean"] for method in display_methods}
    legacy_values = {method: beaver_methods[method]["legacy_worst_norm"]["mean"] for method in display_methods}
    independent_values = {method: independent_methods[method]["worst_marginal_win_rate"] for method in display_methods}
    hard_tex, legacy_tex, independent_tex = decorate(hard_values), decorate(legacy_values), decorate(independent_values)
    rows = [
        f"{LABELS[method]} & {hard_tex[method]} & {legacy_tex[method]} & {independent_tex[method]} \\\\"
        for method in sorted(display_methods, key=hard_values.get, reverse=True)
    ]
    fragment = "\n".join([
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Method & Pair floor & Legacy Worst & Independent wWR$_{\\mathrm{B}}$ \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ])
    (args.output / "frag_stage4_robustness_check.tex").write_text(fragment, encoding="utf-8")

    primary_ci = primary["hierarchical_ci95"]
    independent_ci = independent_ronpo["worst_marginal_win_rate_hierarchical_ci95"]
    report = [
        "# SafeRLHF Stage-4 joint three-seed verification",
        "",
        "No policy was trained, selected, retried, or decoded in this verification.",
        "",
        "## Preregistered decision",
        "",
        f"- RONPO-OS minus IPO hard pairwise floor: **{primary['delta']:+.6f}** "
        f"(hierarchical 95% CI [{primary_ci[0]:+.6f}, {primary_ci[1]:+.6f}]).",
        f"- Per-seed point deltas: " + ", ".join(
            f"seed {seed} {item['delta']:+.6f}" for seed, item in primary["per_seed"].items()
        ) + ".",
        f"- Primary result: **{'PASS' if primary_pass else 'FAIL'}**.",
        "",
        "## Independent evaluator panel",
        "",
        f"- RONPO-OS quality win rate versus Base: "
        f"{independent_ronpo['marginal_win_rates_vs_base']['quality']:.4f}.",
        f"- RONPO-OS safety win rate versus Base: "
        f"{independent_ronpo['marginal_win_rates_vs_base']['safety']:.4f}.",
        f"- Worst marginal win rate: {independent_ronpo['worst_marginal_win_rate']:.4f} "
        f"(hierarchical 95% CI [{independent_ci[0]:.4f}, {independent_ci[1]:.4f}]).",
        f"- Independent non-regression support: **{'PASS' if independent_pass else 'FAIL'}**.",
        "",
        "## Interpretation",
        "",
        (
            "The theory-aligned Stage-4 advantage is supported under the frozen response adversary."
            if primary_pass else
            "The existing normalized-reward lead does not establish a theory-aligned Stage-4 advantage under the frozen response adversary."
        ),
        (
            "The independent quality and safety panel also supports non-regression."
            if independent_pass else
            "The independent quality and safety panel does not support a two-objective non-regression claim."
        ),
        "The legacy normalized numbers are retained only for continuity because they depend on the frozen policy set.",
        "",
        "`spent_sealed_split_touched=false`",
        "",
    ]
    (args.output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    ledger = {
        name: {"path": str(path), "rows": len(read_jsonl(path)), "sha256": sha256(path)}
        for name, path in score_paths.items()
    }
    audit_lines = [
        "# Completion audit",
        "",
        f"- Response pool SHA-256: `{sha256(args.pool)}`.",
        f"- Policies: {len(names)}; prompts: {len(pool)}.",
        "- All compared responses were scored in the frozen per-prompt 28-policy context.",
        "",
        "| Score file | Rows | SHA-256 |",
        "|---|---:|---|",
    ]
    for name, item in ledger.items():
        audit_lines.append(f"| {name} | {item['rows']} | `{item['sha256']}` |")
    audit_lines.extend(["", f"- Primary result: {'PASS' if primary_pass else 'FAIL'}.",
                        f"- Independent support: {'PASS' if independent_pass else 'FAIL'}.",
                        "", "`spent_sealed_split_touched=false`", ""])
    (args.output / "COMPLETION_AUDIT.md").write_text("\n".join(audit_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
