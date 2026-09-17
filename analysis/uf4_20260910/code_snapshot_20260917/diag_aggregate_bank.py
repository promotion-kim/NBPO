"""Turn the bank verdicts into the target-transfer table, on CPU.

For every panel prompt and objective this builds the 8x4 learner-reference
margin matrix and the 4x4 antisymmetric reference-reference margin matrix from
order-averaged verdicts, then evaluates each candidate distribution with
Eq. (transfer-wins) and the finite-bank surplus of Eq. (transfer-surplus) at the
predeclared beta_eval = 0.25.

Uncertainty is a whole-prompt paired bootstrap: one resample of prompts drives
every row, objective, minimum and pairwise difference, so the intervals are
paired across rows. Seed SD is a different quantity and is not produced here.

Nothing in this file refits a target, drops a candidate or renormalizes around a
missing verdict. A (prompt, objective) cell enters the complete case only when
all 38 pairs have both orders parsed; incomplete prompts are counted and also
bounded by putting every missing preference at 0 and at 1.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/work/uf4_20260910")
DIAG = ROOT / "analysis/diag_20260914"
CRITERIA = ("instruction_following", "truthfulness", "honesty", "helpfulness")
BETA_EVAL = 0.25
N_LEARNER, N_REF = 8, 4
TARGETS = [("nbpo", "nash_v1"), ("fixedref", "fixedref_nash_v1"), ("util", "util_l1matched_v1")]


def softmin(margins, mu, beta):
    """-beta log sum_j mu_j exp(-m_j/beta), stable."""
    z = -margins / beta
    m = z.max()
    return -beta * (m + np.log(np.sum(mu * np.exp(z - m))))


def read_verdicts(pattern):
    """(prompt, criterion) -> {(left, right): {order: value}} plus contract counters."""
    cells = defaultdict(lambda: defaultdict(dict))
    checks = {"order_pairs": 0, "order_agree": 0, "order_flip": 0,
              "duplicate_text_pairs": 0, "duplicate_text_value_sum": 0.0,
              "verdicts": 0, "ok": 0}
    for path in sorted(glob.glob(pattern)):
        with open(path) as stream:
            for line in stream:
                r = json.loads(line)
                checks["verdicts"] += 1
                if r["status"] != "ok":
                    continue
                checks["ok"] += 1
                cells[(r["prompt_id"], r["criterion"])][(r["left"], r["right"])][r["order"]] = (
                    r["value_for_left"], r.get("left_sha256"), r.get("right_sha256"))
    for key, pairs in cells.items():
        for pair, byorder in pairs.items():
            if 0 in byorder and 1 in byorder:
                checks["order_pairs"] += 1
                v0, v1 = byorder[0][0], byorder[1][0]
                if v0 == v1:
                    checks["order_agree"] += 1
                else:
                    checks["order_flip"] += 1
                if byorder[0][1] == byorder[0][2]:
                    checks["duplicate_text_pairs"] += 1
                    checks["duplicate_text_value_sum"] += 0.5 * (v0 + v1)
    return cells, checks


def matrices(cells, pid, criterion):
    """(A 8x4, B 4x4, complete) order-averaged margins for one prompt-objective."""
    pairs = cells.get((pid, criterion), {})
    A = np.full((N_LEARNER, N_REF), np.nan)
    B = np.zeros((N_REF, N_REF))
    seen_B = np.zeros((N_REF, N_REF), dtype=bool)
    for (left, right), byorder in pairs.items():
        if 0 not in byorder or 1 not in byorder:
            continue
        value = 0.5 * (byorder[0][0] + byorder[1][0]) - 0.5
        if left.startswith("L"):
            A[int(left[1:]), int(right[1:])] = value
        else:
            a, b = int(left[1:]), int(right[1:])
            B[a, b], B[b, a] = value, -value
            seen_B[a, b] = seen_B[b, a] = True
    need_B = N_REF * (N_REF - 1)
    complete = bool(np.isfinite(A).all() and seen_B.sum() == need_B)
    return A, B, complete


def row_values(A_list, B_list, p_list, mu):
    """Direct wins and finite-bank surplus per objective, averaged over prompts."""
    W, V, D = [], [], []
    for A, B, p in zip(A_list, B_list, p_list):
        W.append(0.5 + float(p @ A @ mu))
        V.append(softmin(A.T @ p, mu, BETA_EVAL))
        D.append(softmin(B.T @ mu, mu, BETA_EVAL))
    return np.array(W), np.array(V), np.array(D)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", default=str(DIAG / "bank4/verdicts/*.jsonl"))
    ap.add_argument("--fresh-dir", default=str(DIAG / "fresh_verdicts"))
    ap.add_argument("--neural", default=str(DIAG / "neural_pools.json"))
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--out", default=str(DIAG / "target_transfer.json"))
    args = ap.parse_args()

    panel = json.loads((DIAG / "panel_dev200.json").read_text())
    ids = panel["panel_prompt_ids"]
    cells, checks = read_verdicts(args.bank)

    # ---- candidate distributions, indexed by the target tensors' own prompt order
    dists = {}
    for name, target in TARGETS:
        meta = json.loads((ROOT / "targets" / target / "dev/tensor/meta.json").read_text())
        index = {pid: k for k, pid in enumerate(meta["prompt_ids"])}
        pi_star = np.load(ROOT / "targets" / target / "dev/solver/pi_star.npz")["pi"]
        pi_t = np.load(ROOT / "targets" / target / "dev/solver/pi_t.npz")["pi"]
        dists[name] = {pid: pi_star[index[pid]] for pid in ids if pid in index}
        if name == "nbpo":
            dists["source"] = {pid: pi_t[index[pid]] for pid in ids if pid in index}
    neural_path = Path(args.neural)
    if neural_path.exists():
        for name, table in json.loads(neural_path.read_text()).items():
            dists["neural_" + name] = {pid: np.array(v) for pid, v in table.items()}

    mu = np.full(N_REF, 1.0 / N_REF)
    rows, complete_ids = {}, {}
    for criterion in CRITERIA:
        usable = [pid for pid in ids if matrices(cells, pid, criterion)[2]]
        complete_ids[criterion] = usable
        mats = [matrices(cells, pid, criterion)[:2] for pid in usable]
        A_list = [m[0] for m in mats]
        B_list = [m[1] for m in mats]
        for name, table in dists.items():
            keep = [k for k, pid in enumerate(usable) if pid in table]
            if len(keep) != len(usable):
                continue
            p_list = [table[pid] for pid in usable]
            W, V, D = row_values(A_list, B_list, p_list, mu)
            rows.setdefault(name, {})[criterion] = {
                "n": len(usable), "win_rate": float(W.mean()),
                "surplus": float((V - D).mean()),
                "per_prompt_w": W.tolist(), "per_prompt_s": (V - D).tolist(),
            }

    # ---- fresh rows: direct wins only, one draw per prompt
    fresh_rows = {}
    for path in sorted(glob.glob(str(Path(args.fresh_dir) / "*.jsonl"))):
        arm = Path(path).stem
        fcells, fchecks = read_verdicts(path)
        for criterion in CRITERIA:
            values, used = [], []
            for pid in ids:
                pairs = fcells.get((pid, criterion), {})
                per_ref = []
                for (left, right), byorder in pairs.items():
                    if 0 in byorder and 1 in byorder:
                        per_ref.append(0.5 * (byorder[0][0] + byorder[1][0]))
                if len(per_ref) == N_REF:
                    values.append(float(np.mean(per_ref)))
                    used.append(pid)
            if values:
                fresh_rows.setdefault(arm, {})[criterion] = {
                    "n": len(values), "win_rate": float(np.mean(values)),
                    "per_prompt_w": values, "prompt_ids": used,
                    "surplus": None, "checks": fchecks,
                }

    # ---- bounds: every unresolved preference at 0 and at 1, on all 200 prompts
    bounds = {}
    for criterion in CRITERIA:
        raw = {pid: matrices(cells, pid, criterion) for pid in ids}
        for name, table in dists.items():
            lo_w, hi_w, lo_s, hi_s, used = [], [], [], [], 0
            for pid in ids:
                if pid not in table:
                    continue
                A, B, _ = raw[pid]
                p = np.asarray(table[pid])
                A_lo = np.where(np.isfinite(A), A, -0.5)
                A_hi = np.where(np.isfinite(A), A, 0.5)
                lo_w.append(0.5 + float(p @ A_lo @ mu))
                hi_w.append(0.5 + float(p @ A_hi @ mu))
                # B is filled by antisymmetry; unresolved reference pairs stay 0,
                # so its bounds move with the same +/-0.5 envelope on zero entries
                zero = (B == 0) & ~np.eye(N_REF, dtype=bool)
                B_lo = np.where(zero, -0.5, B)
                B_hi = np.where(zero, 0.5, B)
                lo_s.append(softmin(A_lo.T @ p, mu, BETA_EVAL) - softmin(B_hi.T @ mu, mu, BETA_EVAL))
                hi_s.append(softmin(A_hi.T @ p, mu, BETA_EVAL) - softmin(B_lo.T @ mu, mu, BETA_EVAL))
                used += 1
            if used:
                bounds.setdefault(name, {})[criterion] = {
                    "n_planned": used,
                    "win_rate_lower": float(np.mean(lo_w)),
                    "win_rate_upper": float(np.mean(hi_w)),
                    "surplus_lower": float(np.mean(lo_s)),
                    "surplus_upper": float(np.mean(hi_s)),
                }

    # ---- judge contract on this panel: actual self-pairs and order sensitivity
    # ---- whole-prompt paired bootstrap over the complete intersection
    inter = sorted(set.intersection(*[set(complete_ids[c]) for c in CRITERIA])) if cells else []
    rng = np.random.default_rng(args.seed)
    boot = {}
    if inter:
        pos = {c: {pid: k for k, pid in enumerate(complete_ids[c])} for c in CRITERIA}
        draws = rng.integers(0, len(inter), size=(args.replicates, len(inter)))
        for name, per_c in rows.items():
            boot[name] = {}
            for criterion in CRITERIA:
                w = np.array(per_c[criterion]["per_prompt_w"])
                s = np.array(per_c[criterion]["per_prompt_s"])
                take = np.array([pos[criterion][pid] for pid in inter])
                bw = w[take][draws].mean(axis=1)
                bs = s[take][draws].mean(axis=1)
                boot[name][criterion] = {"w": bw, "s": bs}
        summary = {}
        for name in rows:
            mins = np.min(np.stack([boot[name][c]["s"] for c in CRITERIA]), axis=0)
            summary[name] = {"min_surplus_ci95": [float(np.percentile(mins, 2.5)),
                                                  float(np.percentile(mins, 97.5))],
                             "min_surplus_point": float(min(
                                 float(np.mean(np.array(rows[name][c]["per_prompt_s"])
                                               [[pos[c][pid] for pid in inter]]))
                                 for c in CRITERIA))}
            for criterion in CRITERIA:
                bw = boot[name][criterion]["w"]
                summary[name][criterion] = {
                    "win_rate_common": float(np.mean(np.array(rows[name][criterion]["per_prompt_w"])
                                                     [[pos[criterion][pid] for pid in inter]])),
                    "ci95": [float(np.percentile(bw, 2.5)), float(np.percentile(bw, 97.5))],
                }
        diffs = {}
        for a, b in (("nbpo", "source"), ("nbpo", "util"), ("nbpo", "fixedref")):
            if a in boot and b in boot:
                diffs["%s_minus_%s" % (a, b)] = {
                    c: {"point": summary[a][c]["win_rate_common"] - summary[b][c]["win_rate_common"],
                        "ci95": [float(np.percentile(boot[a][c]["w"] - boot[b][c]["w"], 2.5)),
                                 float(np.percentile(boot[a][c]["w"] - boot[b][c]["w"], 97.5))]}
                    for c in CRITERIA}
    else:
        summary, diffs = {}, {}

    # ---- fresh rows: per-arm interval and the paired NBPO-minus-utilitarian stage
    fresh_ci, fresh_diffs = {}, {}
    fresh_per_prompt = {}
    for path in sorted(glob.glob(str(Path(args.fresh_dir) / "*.jsonl"))):
        arm = Path(path).stem
        fcells, _ = read_verdicts(path)
        per_c = {}
        for criterion in CRITERIA:
            vals = {}
            for pid in ids:
                pairs = fcells.get((pid, criterion), {})
                got = [0.5 * (v[0][0] + v[1][0]) for v in pairs.values()
                       if 0 in v and 1 in v]
                if len(got) == N_REF:
                    vals[pid] = float(np.mean(got))
            per_c[criterion] = vals
        fresh_per_prompt[arm] = per_c
        shared = sorted(set.intersection(*[set(per_c[c]) for c in CRITERIA])) or []
        if not shared:
            continue
        d = rng.integers(0, len(shared), size=(args.replicates, len(shared)))
        entry, mins = {}, []
        for criterion in CRITERIA:
            x = np.array([per_c[criterion][pid] for pid in shared])
            b = x[d].mean(axis=1)
            entry[criterion] = {"win_rate": float(x.mean()),
                                "ci95": [float(np.percentile(b, 2.5)),
                                         float(np.percentile(b, 97.5))]}
            mins.append(b)
        w = np.min(np.stack(mins), axis=0)
        entry["w_min"] = {"point": float(np.min([entry[c]["win_rate"] for c in CRITERIA])),
                          "ci95": [float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))]}
        entry["n"] = len(shared)
        fresh_ci[arm] = entry

    a_arm, b_arm = "nbpo_mse_s42", "util_mse_s42"
    if a_arm in fresh_per_prompt and b_arm in fresh_per_prompt:
        A, B = fresh_per_prompt[a_arm], fresh_per_prompt[b_arm]
        shared = sorted(set.intersection(*[set(A[c]) for c in CRITERIA],
                                        *[set(B[c]) for c in CRITERIA]))
        if shared:
            d = rng.integers(0, len(shared), size=(args.replicates, len(shared)))
            fresh_diffs = {"n": len(shared)}
            for criterion in CRITERIA:
                x = np.array([A[criterion][pid] for pid in shared])
                y = np.array([B[criterion][pid] for pid in shared])
                b = (x - y)[d].mean(axis=1)
                fresh_diffs[criterion] = {
                    "point": float((x - y).mean()),
                    "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}

    pool_diffs = {}
    if "neural_nbpo_mse_s42" in boot and "neural_util_mse_s42" in boot:
        for criterion in CRITERIA:
            b = (boot["neural_nbpo_mse_s42"][criterion]["w"]
                 - boot["neural_util_mse_s42"][criterion]["w"])
            pool_diffs[criterion] = {
                "point": (summary["neural_nbpo_mse_s42"][criterion]["win_rate_common"]
                          - summary["neural_util_mse_s42"][criterion]["win_rate_common"]),
                "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}

    payload = {
        "panel_sha256": panel["panel_sha256"],
        "beta_eval": BETA_EVAL,
        "reference_bank_size": N_REF,
        "replicates": args.replicates,
        "bootstrap_seed": args.seed,
        "common_prompts_all_objectives": len(inter),
        "complete_per_objective": {c: len(complete_ids.get(c, [])) for c in CRITERIA},
        "planned_prompts": len(ids),
        "contract_checks": checks,
        "rows": {k: {c: {q: v[c][q] for q in ("n", "win_rate", "surplus")} for c in v}
                 for k, v in rows.items()},
        "fresh_rows": {k: {c: {q: v[c][q] for q in ("n", "win_rate")} for c in v}
                       for k, v in fresh_rows.items()},
        "bootstrap": summary,
        "bounds_all_planned_prompts": bounds,
        "paired_differences": diffs,
        "fresh_intervals": fresh_ci,
        "fresh_nbpo_minus_util": fresh_diffs,
        "pool_nbpo_minus_util": pool_diffs,
        "note": ("Finite-bank surrogate on four common reference responses; not a population "
                 "game-value estimate. Fresh rows are one stochastic draw per prompt and carry "
                 "no surplus."),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in
                      ("common_prompts_all_objectives", "complete_per_objective",
                       "contract_checks", "rows", "fresh_rows")}, indent=2)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
