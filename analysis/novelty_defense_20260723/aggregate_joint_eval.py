#!/usr/bin/env python3
"""Aggregate the frozen joint RM pass and preregistered paired endpoints."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp

OBJECTIVES = ["skywork", "athene", "armo"]


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ci_mean(values: np.ndarray, indices: np.ndarray) -> list[float]:
    boot = values[indices].mean(axis=1)
    return [float(x) for x in np.percentile(boot, [2.5, 97.5])]


def load_joint(scored: Path, policy_names: list[str]) -> tuple[list[str], list[np.ndarray], list[list[str]]]:
    rows_by_obj = {obj: read_jsonl(scored / f"joint_{obj}.jsonl") for obj in OBJECTIVES}
    prompts = [row["prompt"] for row in rows_by_obj[OBJECTIVES[0]]]
    if len(prompts) != 647 or len(set(prompts)) != 647:
        raise ValueError(f"expected 647 unique prompts, found {len(prompts)}/{len(set(prompts))}")
    by_obj = {obj: {row["prompt"]: row for row in rows} for obj, rows in rows_by_obj.items()}
    values = []
    names_by_prompt = []
    for prompt in prompts:
        names = by_obj[OBJECTIVES[0]][prompt]["response_model_names"]
        if names[: len(policy_names)] != policy_names:
            raise ValueError(f"policy order mismatch at prompt {prompt[:80]!r}")
        names_by_prompt.append(names)
        obj_values = []
        for obj in OBJECTIVES:
            row = by_obj[obj][prompt]
            if row["response_model_names"] != names:
                raise ValueError(f"response order mismatch for {obj}")
            obj_values.append(row["all_rm_scores"])
        values.append(obj_values)
    return prompts, [np.asarray(row, dtype=float) for row in values], names_by_prompt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scored", required=True)
    p.add_argument("--lock", required=True)
    p.add_argument("--generations", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = p.parse_args()

    lock = json.loads(Path(args.lock).read_text())
    policies = lock["normalization_policies"]
    prompts, all_scores, names_by_prompt = load_joint(Path(args.scored), policies)
    n = len(all_scores)
    k = len(OBJECTIVES)
    m = len(policies)
    policy_scores = np.stack([row[:, :m] for row in all_scores])

    lo = policy_scores.min(axis=2, keepdims=True)
    hi = policy_scores.max(axis=2, keepdims=True)
    span = hi - lo
    norm = np.divide(policy_scores - lo, span, out=np.full_like(policy_scores, 0.5), where=span > 0)
    prompt_worst = norm.min(axis=1)
    prompt_avg = norm.mean(axis=1)

    beta = np.asarray([lock["pairwise_floor"]["beta"][obj] for obj in OBJECTIVES])
    kappa = float(lock["pairwise_floor"]["kappa"])
    hard = np.empty((n, m))
    soft = np.empty((n, m))
    for x in range(n):
        opponents = all_scores[x]
        for mi in range(m):
            costs = expit(beta[:, None] * (policy_scores[x, :, mi, None] - opponents))
            hard[x, mi] = costs.min()
            soft[x, mi] = -kappa * (logsumexp(-costs / kappa) - np.log(costs.size))

    rng = np.random.default_rng(42)
    indices = rng.integers(0, n, size=(2000, n))
    base = policies.index("base")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    token_means = {}
    for name in policies:
        rows = json.loads((Path(args.generations) / name / "output_42.json").read_text())
        counts = []
        texts = [row["generated_text"] for row in rows]
        for start in range(0, len(texts), 64):
            counts.extend(len(ids) for ids in tokenizer(texts[start:start + 64], add_special_tokens=False)["input_ids"])
        token_means[name] = float(np.mean(counts))

    per_policy = {}
    for mi, name in enumerate(policies):
        objective_means = norm[:, :, mi].mean(axis=0)
        raw_means = policy_scores[:, :, mi].mean(axis=0)
        win = np.where(policy_scores[:, :, mi] > policy_scores[:, :, base], 1.0,
                       np.where(policy_scores[:, :, mi] < policy_scores[:, :, base], 0.0, 0.5))
        win_obj = win.mean(axis=0)
        per_policy[name] = {
            "normalized_objectives": dict(zip(OBJECTIVES, map(float, objective_means))),
            "raw_objectives": dict(zip(OBJECTIVES, map(float, raw_means))),
            "mean_objective_norm_score": float(prompt_avg[:, mi].mean()),
            "mean_prompt_worst_norm_score": float(prompt_worst[:, mi].mean()),
            "worst_norm_ci95": ci_mean(prompt_worst[:, mi], indices),
            "mean_win_rate_vs_base": float(win_obj.mean()),
            "min_win_rate_vs_base": float(win_obj.min()),
            "win_rate_objectives": dict(zip(OBJECTIVES, map(float, win_obj))),
            "pairwise_floor": float(hard[:, mi].mean()),
            "pairwise_floor_ci95": ci_mean(hard[:, mi], indices),
            "soft_pairwise_floor": float(soft[:, mi].mean()),
            "soft_pairwise_floor_ci95": ci_mean(soft[:, mi], indices),
            "mean_tokens": token_means[name],
        }

    best_soft = float(soft.mean(axis=0).max())
    for name, values in per_policy.items():
        values["finite_set_duality_gap_proxy"] = best_soft - values["soft_pairwise_floor"]

    contrasts = {}
    for left, right in lock["primary_contrasts"]:
        li, ri = policies.index(left), policies.index(right)
        contrasts[f"{left}_minus_{right}"] = {
            "pairwise_floor_delta": float((hard[:, li] - hard[:, ri]).mean()),
            "pairwise_floor_delta_ci95": ci_mean(hard[:, li] - hard[:, ri], indices),
            "worst_norm_delta": float((prompt_worst[:, li] - prompt_worst[:, ri]).mean()),
            "worst_norm_delta_ci95": ci_mean(prompt_worst[:, li] - prompt_worst[:, ri], indices),
            "avg_norm_delta": float((prompt_avg[:, li] - prompt_avg[:, ri]).mean()),
            "avg_norm_delta_ci95": ci_mean(prompt_avg[:, li] - prompt_avg[:, ri], indices),
        }

    out = Path(args.output)
    (out / "per_policy_scores").mkdir(parents=True, exist_ok=True)
    paired = {
        "num_prompts": n,
        "normalization_policies": policies,
        "bootstrap": {"resamples": 2000, "seed": 42},
        "per_policy": per_policy,
        "primary_contrasts": contrasts,
        "note": "All normalized values use the frozen 15-policy joint context.",
    }
    floor = {
        "num_prompts": n,
        "beta": dict(zip(OBJECTIVES, map(float, beta))),
        "kappa": kappa,
        "formula": "mean_x min_k,a sigmoid(beta_k*(raw_r_k(y)-raw_r_k(a)))",
        "per_policy": {name: {key: val for key, val in values.items() if "floor" in key or "duality" in key}
                       for name, values in per_policy.items()},
        "primary_contrasts": {name: {key: val for key, val in values.items() if "floor" in key}
                              for name, values in contrasts.items()},
    }
    (out / "paired_summary.json").write_text(json.dumps(paired, indent=2) + "\n")
    (out / "floor_summary.json").write_text(json.dumps(floor, indent=2) + "\n")
    with (out / "per_policy_scores" / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", *OBJECTIVES, "avg", "worst", "worst_ci_low", "worst_ci_high",
                         "wr_b", "wwr_b", "pairwise_floor", "soft_floor", "gap_proxy", "mean_tokens"])
        for name in policies:
            v = per_policy[name]
            writer.writerow([name, *[v["normalized_objectives"][obj] for obj in OBJECTIVES],
                             v["mean_objective_norm_score"], v["mean_prompt_worst_norm_score"],
                             *v["worst_norm_ci95"], v["mean_win_rate_vs_base"], v["min_win_rate_vs_base"],
                             v["pairwise_floor"], v["soft_pairwise_floor"],
                             v["finite_set_duality_gap_proxy"], v["mean_tokens"]])
    for mi, name in enumerate(policies):
        prompt_rows = []
        for xi, prompt in enumerate(prompts):
            prompt_rows.append({
                "prompt_id": hashlib.sha256(prompt.encode()).hexdigest(),
                "raw": {obj: float(policy_scores[xi, ki, mi]) for ki, obj in enumerate(OBJECTIVES)},
                "normalized": {obj: float(norm[xi, ki, mi]) for ki, obj in enumerate(OBJECTIVES)},
                "prompt_worst": float(prompt_worst[xi, mi]),
                "prompt_average": float(prompt_avg[xi, mi]),
                "pairwise_floor": float(hard[xi, mi]),
                "soft_pairwise_floor": float(soft[xi, mi]),
            })
        (out / "per_policy_scores" / f"{name}.json").write_text(
            json.dumps({"policy": name, "rows": prompt_rows}, separators=(",", ":")) + "\n"
        )
    print(json.dumps({"prompts": n, "policies": m, "output": str(out)}))


if __name__ == "__main__":
    main()
