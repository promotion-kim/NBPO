"""Separate target difference from neural fit error on the one development panel.

On the frozen 200 development prompts, with the uniform measure over each
prompt's 28 unordered candidate pairs:

    h*_a(i,j) = log(p*_{a,i}/p_{t,i}) - log(p*_{a,j}/p_{t,j})
    h_a(i,j)  = delta_{a,i} - delta_{a,j},  delta_{a,i} = log pi_a(y_i) - log pi_t(y_i)
    e_a       = h_a - h*_a

and the identity h_a - h_b = (h*_a - h*_b) + (e_a - e_b) is measured term by
term. The ratio of these scales is not a power calculation and a large fit error
alone does not prove the method contrast is lost; both are reported so the
reader can see which term carries the difference.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/work/uf4_20260910")
DIAG = ROOT / "analysis/diag_20260914"
PAIRS = [(i, j) for i in range(8) for j in range(i + 1, 8)]
ARMS = {"nbpo": ("nbpo_mse_s42", "nash_v1"),
        "util": ("util_mse_s42", "util_l1matched_v1"),
        "fixedref": ("fixedref_mse_s42", "fixedref_nash_v1")}
FLOOR = 1e-12


def load(target):
    meta = json.loads((ROOT / "targets" / target / "dev/tensor/meta.json").read_text())
    index = {pid: k for k, pid in enumerate(meta["prompt_ids"])}
    pi_star = np.load(ROOT / "targets" / target / "dev/solver/pi_star.npz")["pi"]
    pi_t = np.load(ROOT / "targets" / target / "dev/solver/pi_t.npz")["pi"]
    return index, pi_star, pi_t


def pair_vector(values):
    return np.array([values[i] - values[j] for i, j in PAIRS])


def main():
    panel = json.loads((DIAG / "panel_dev200.json").read_text())
    ids = panel["panel_prompt_ids"]
    ratios = json.loads((DIAG / "pool_log_ratios.json").read_text())

    per_arm = {}
    for key, (arm, target) in ARMS.items():
        if arm not in ratios:
            continue
        index, pi_star, pi_t = load(target)
        h_star, e, tv_ref = {}, {}, {}
        for pid in ids:
            if pid not in index or pid not in ratios[arm]:
                continue
            k = index[pid]
            star = np.clip(pi_star[k], FLOOR, None)
            src = np.clip(pi_t[k], FLOOR, None)
            h_star[pid] = pair_vector(np.log(star) - np.log(src))
            e[pid] = pair_vector(np.asarray(ratios[arm][pid])) - h_star[pid]
            tv_ref[pid] = star
        per_arm[key] = {"h_star": h_star, "e": e, "p_star": tv_ref, "arm": arm,
                        "target": target}

    def rms(chunks):
        flat = np.concatenate([np.asarray(v).ravel() for v in chunks])
        return float(np.sqrt(np.mean(flat ** 2)))

    out = {}
    for label, a, b in (("nbpo_vs_util", "nbpo", "util"),
                        ("nbpo_vs_fixedref", "nbpo", "fixedref")):
        if a not in per_arm or b not in per_arm:
            continue
        shared = [pid for pid in ids if pid in per_arm[a]["e"] and pid in per_arm[b]["e"]]
        if not shared:
            continue
        out[label] = {
            "n": len(shared),
            "mean_tv": float(np.mean([0.5 * np.abs(per_arm[a]["p_star"][pid]
                                                   - per_arm[b]["p_star"][pid]).sum()
                                      for pid in shared])),
            "target_diff_rms": rms([per_arm[a]["h_star"][pid] - per_arm[b]["h_star"][pid]
                                    for pid in shared]),
            "nbpo_error_rms": rms([per_arm[a]["e"][pid] for pid in shared]),
            "error_diff_rms": rms([per_arm[a]["e"][pid] - per_arm[b]["e"][pid]
                                   for pid in shared]),
            "control_error_rms": rms([per_arm[b]["e"][pid] for pid in shared]),
            "target_rms_a": rms([per_arm[a]["h_star"][pid] for pid in shared]),
            "target_rms_b": rms([per_arm[b]["h_star"][pid] for pid in shared]),
            "fitted_diff_rms": rms([(per_arm[a]["h_star"][pid] + per_arm[a]["e"][pid])
                                    - (per_arm[b]["h_star"][pid] + per_arm[b]["e"][pid])
                                    for pid in shared]),
            "arms": [per_arm[a]["arm"], per_arm[b]["arm"]],
            "targets": [per_arm[a]["target"], per_arm[b]["target"]],
        }
    (DIAG / "target_signal.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
