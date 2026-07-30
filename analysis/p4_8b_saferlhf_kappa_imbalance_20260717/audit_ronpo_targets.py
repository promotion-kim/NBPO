#!/usr/bin/env python3
"""Fail-closed P4 audit for the static RONPO target families."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from datasets import load_from_disk


def tag(kappa: float) -> str:
    return f"{kappa:g}".replace(".", "p")


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() == 0.0 or y.std() == 0.0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--kappa-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.kappa_lock.read_text(encoding="utf-8"))
    kappas = [float(item["selected_kappa"]) for item in lock["selected"]]
    loaded = load_from_disk(str(args.input_dir))
    splits = list(loaded.keys()) if hasattr(loaded, "keys") else [None]
    rows = []
    for split in splits:
        rows.extend(loaded[split] if split is not None else loaded)
    if not rows:
        raise RuntimeError("empty target dataset")
    uniform = np.asarray([float(row["target_uniform"]) for row in rows])
    top = {k: np.asarray([float(row[f"target_topmass_k{tag(k)}"]) for row in rows]) for k in kappas}
    os_values = {k: np.asarray([float(row[f"target_os_k{tag(k)}"]) for row in rows]) for k in kappas}
    full = {k: np.asarray([float(row[f"target_fullexp_k{tag(k)}"]) for row in rows]) for k in kappas}
    reference_top = top[kappas[0]]
    top_identical = {
        f"{k:g}": {
            "fraction_differing_from_first": float(np.mean(values != reference_top)),
            "max_abs_difference_from_first": float(np.max(np.abs(values - reference_top))),
        }
        for k, values in top.items()
    }
    ordered = sorted(kappas)
    curve = []
    for kappa in ordered:
        f = full[kappa]
        curve.append({
            "kappa": kappa,
            "full_exp_corr_topmass": corr(f, reference_top),
            "full_exp_corr_uniform": corr(f, uniform),
            "full_exp_mean_abs": float(np.abs(f).mean()),
            "os_mean_abs": float(np.abs(os_values[kappa]).mean()),
            "os_vs_full_fraction_differing": float(np.mean(os_values[kappa] != f)),
            "os_vs_topmass_fraction_differing": float(np.mean(os_values[kappa] != reference_top)),
            "full_vs_topmass_fraction_differing": float(np.mean(f != reference_top)),
            "full_vs_uniform_fraction_differing": float(np.mean(f != uniform)),
        })
    top_ok = all(value["fraction_differing_from_first"] == 0.0 for value in top_identical.values())
    corr_top = [row["full_exp_corr_topmass"] for row in curve]
    corr_uniform = [row["full_exp_corr_uniform"] for row in curve]
    monotone_top = all(b <= a + 1e-12 for a, b in zip(corr_top, corr_top[1:]))
    monotone_uniform = all(b >= a - 1e-12 for a, b in zip(corr_uniform, corr_uniform[1:]))
    interior = [row for row in curve if row["os_vs_topmass_fraction_differing"] > 0.0 and row["full_vs_topmass_fraction_differing"] > 0.0]
    payload = {
        "status": "pass" if top_ok and monotone_top and monotone_uniform and interior else "fail",
        "scope": "reward-blind target audit on shared precomputed rows before any training",
        "rows": len(rows),
        "kappas_sorted": ordered,
        "topmass_exact_kappa_invariance": top_identical,
        "curve": curve,
        "assertions": {
            "topmass_identical_all_kappas": top_ok,
            "full_exp_corr_topmass_nonincreasing": monotone_top,
            "full_exp_corr_uniform_nondecreasing": monotone_uniform,
            "interior_target_families_nonidentical": bool(interior),
        },
        "static_adversary_note": "sigma is computed once from uniform-pool costs; this audit does not claim adaptive OMD dynamics.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
