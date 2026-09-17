#!/usr/bin/env python3
"""Section F: audit the frozen protocol on held-out real pairs.

Unlike the calibration set these pairs have no ground truth, so the questions
are about *self-consistency and reliability* rather than accuracy. The gates
below are evaluated exactly as written; nothing here is tuned after seeing them,
and a failing gate is reported as failing.

Two metrics are reported that look similar and are not:

``raw_all_pair_hard_swap_consistency``
    agreement over every pair both orders decided, whatever the confidence.
    This is the v2 quantity. It is a **diagnostic**, not a gate, because on a
    pool of same-policy samples a large fraction of pairs are genuine near-ties
    where disagreement is the honest answer.

``confident_decisive_swap_consistency``
    the gate. Restricted to pairs the protocol is confident about -- both orders
    decisive by a margin and tie mass below half. A protocol is allowed to be
    unsure; it is not allowed to be sure and wrong.

Cycles are counted on **reliable edges only** (confident, non-tie, adjudicated
to resolution), since an edge the protocol never committed to says nothing about
transitivity. The soft cycle mass uses every edge and is reported beside it.

Split-half: on the operational adaptive path most pairs carry a single template,
so a template split is only available on the adjudicated subset. The order split
(forward estimate vs reverse estimate) is available everywhere and is reported
as such -- the two are labelled differently because they measure different
things, and the template split is the one Section G needs.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

FORWARD, REVERSE = "learner_first", "comparator_first"
DECISIVE_MARGIN = 0.1
CONFIDENT_TIE_MASS = 0.5
HIGH_ENTROPY = 0.90

GATES = {
    "invalid_rate_max": 0.002,
    "abs_signed_position_bias_max": 0.03,
    "confident_decisive_swap_consistency_min": 0.85,
    "split_half_spearman_min": 0.75,
    "unresolved_uncertainty_rate_max": 0.25,
}


def read_jsonl(p: Path):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else None


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0 + 1
            i = j + 1
        return r
    return pearson(rank(xs), rank(ys))


def bootstrap_ci(vals, n=2000, seed=20260907):
    if len(vals) < 3:
        return (None, None)
    import random
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        s = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
        means.append(sum(s) / len(s))
    means.sort()
    return (means[int(0.025 * n)], means[int(0.975 * n)])


def half_delta(obs, tids):
    f = [o for o in obs if o["template_id"] in tids and o["presentation_order"] == FORWARD]
    r = [o for o in obs if o["template_id"] in tids and o["presentation_order"] == REVERSE]
    if not f or not r:
        return None
    return 0.5 * (sum(o["semantic_score"] for o in f) / len(f)
                  + sum(o["semantic_score"] for o in r) / len(r)) - 0.5


def bt_deviance(edges):
    """Deviance of a Bradley-Terry fit to one prompt's edges, by simple MM iteration."""
    nodes = sorted({n for e in edges for n in e[:2]})
    if len(nodes) < 3:
        return None
    idx = {n: i for i, n in enumerate(nodes)}
    w = [1.0] * len(nodes)
    for _ in range(200):
        new = []
        for i, n in enumerate(nodes):
            num = den = 0.0
            for a, b, p in edges:
                if a == n:
                    num += p
                    den += 1.0 / (w[i] + w[idx[b]])
                elif b == n:
                    num += 1.0 - p
                    den += 1.0 / (w[i] + w[idx[a]])
            new.append(num / den if den > 0 else w[i])
        s = sum(new) / len(new)
        w = [x / s for x in new]
    dev = 0.0
    for a, b, p in edges:
        q = w[idx[a]] / (w[idx[a]] + w[idx[b]])
        for obs_p, pred in ((p, q), (1 - p, 1 - q)):
            if obs_p > 0 and pred > 0:
                dev += 2 * obs_p * math.log(obs_p / pred)
    return dev / max(1, len(edges))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--control-calibration", type=Path, default=None,
                    help="calibration_results.json, for the control-based gates")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    meta = {r["pair_id"]: r for r in read_jsonl(args.pairs)}
    rows = read_jsonl(args.results)
    for r in rows:
        m = meta.get(r["pair_id"], {})
        r.update({k: m.get(k) for k in ("pool", "prompt_id", "learner_id",
                                        "comparator_id")})
        r["len_diff"] = len(m.get("response_a") or "") - len(m.get("response_b") or "")

    # Keep the invalid rows: the invalid RATE is a hard gate, and computing it
    # over a list that has already been filtered to valid rows makes it
    # identically zero. That is exactly the bug this line replaces.
    all_by_obj = defaultdict(list)
    by_obj = defaultdict(list)
    for r in rows:
        all_by_obj[r["objective"]].append(r)
        if r.get("valid"):
            by_obj[r["objective"]].append(r)

    per_obj, deltas_by_obj = {}, {}
    for obj, rs in sorted(by_obj.items()):
        n = len(rs)
        both_dec = [r for r in rs
                    if abs(r["semantic_score_forward"] - 0.5) > DECISIVE_MARGIN
                    and abs(r["semantic_score_reverse"] - 0.5) > DECISIVE_MARGIN]
        agree = [r for r in both_dec
                 if (r["semantic_score_forward"] - 0.5) * (r["semantic_score_reverse"] - 0.5) > 0]
        confident = [r for r in both_dec if r["mean_tie_probability"] < CONFIDENT_TIE_MASS]
        conf_agree = [r for r in confident
                      if (r["semantic_score_forward"] - 0.5) * (r["semantic_score_reverse"] - 0.5) > 0]
        bias_vals = [r["semantic_score_forward"] - r["semantic_score_reverse"] for r in rs]
        bias = sum(bias_vals) / max(1, n)
        lo, hi = bootstrap_ci(bias_vals)
        d = [r["delta"] for r in rs]
        deltas_by_obj[obj] = {r["pair_id"]: r["delta"] for r in rs}
        q = statistics.quantiles(d, n=10) if len(d) > 10 else []

        of, orv = [], []
        ht, h1, h2 = [], [], []
        for r in rs:
            obs = r.get("observations") or []
            f = [o for o in obs if o["presentation_order"] == FORWARD]
            v = [o for o in obs if o["presentation_order"] == REVERSE]
            if f and v:
                # `semantic_score` is ALREADY the learner's win probability in both
                # orders -- semantic_score() does the slot conversion -- so each
                # order's independent estimate of Delta is (score - 0.5). Writing
                # the reverse one as (0.5 - score) double-flips it and turns the
                # reliability correlation negative, which is how this was caught.
                of.append(sum(o["semantic_score"] for o in f) / len(f) - 0.5)
                orv.append(sum(o["semantic_score"] for o in v) / len(v) - 0.5)
            tids = sorted({o["template_id"] for o in obs})
            if len(tids) >= 2:
                a, b = half_delta(obs, {tids[0]}), half_delta(obs, set(tids[1:]))
                if a is not None and b is not None:
                    h1.append(a)
                    h2.append(b)
                    ht.append(r["pair_id"])

        per_obj[obj] = {
            "n_pairs": n,
            "n_pairs_attempted": len(all_by_obj[obj]),
            "invalid_rate": (sum(1 for r in all_by_obj[obj] if not r.get("valid"))
                             / max(1, len(all_by_obj[obj]))),
            "n_invalid_pairs": sum(1 for r in all_by_obj[obj] if not r.get("valid")),
            "adjudicated_rate": sum(1 for r in rs if r.get("adjudicated")) / max(1, n),
            "unresolved_uncertainty_rate": sum(1 for r in rs
                                               if r.get("unresolved_uncertainty")) / max(1, n),
            "raw_all_pair_hard_swap_consistency": (len(agree) / len(both_dec)
                                                   if both_dec else None),
            "confident_decisive_swap_consistency": (len(conf_agree) / len(confident)
                                                    if confident else None),
            "decisive_contradiction_rate": ((len(both_dec) - len(agree)) / len(both_dec)
                                            if both_dec else None),
            "n_both_orders_decisive": len(both_dec),
            "n_confident_decisive": len(confident),
            "signed_position_bias": bias,
            "position_bias_ci95": [lo, hi],
            "response_length_bias": pearson([r["len_diff"] for r in rs], d),
            "mean_abs_delta": sum(abs(x) for x in d) / max(1, len(d)),
            "abs_delta_deciles": [round(abs(x), 5) for x in q],
            "confident_tie_rate": sum(1 for r in rs
                                      if r["mean_tie_probability"] >= CONFIDENT_TIE_MASS
                                      and abs(r["delta"]) <= 0.05) / max(1, n),
            "high_entropy_rate": sum(1 for r in rs
                                     if r["mean_normalized_entropy"] > HIGH_ENTROPY) / max(1, n),
            "split_half_order_pearson": pearson(of, orv),
            "split_half_order_spearman": spearman(of, orv),
            "split_half_template_pearson": pearson(h1, h2) if h1 else None,
            "split_half_template_spearman": spearman(h1, h2) if h1 else None,
            "n_split_half_template_pairs": len(h1),
        }

    # objective-label correlation
    corr = {}
    for a, b in itertools.combinations(sorted(deltas_by_obj), 2):
        shared = set(k.replace(f"|{a}|", "|") for k in deltas_by_obj[a]) & \
                 set(k.replace(f"|{b}|", "|") for k in deltas_by_obj[b])
        xs, ys = [], []
        for k in shared:
            ka = k.replace("|", f"|{a}|", 1) if False else None
        # rebuild by (pool, prompt, learner, comparator)
        ma = {tuple(k.split("|")[0:2] + k.split("|")[3:]): v for k, v in deltas_by_obj[a].items()}
        mb = {tuple(k.split("|")[0:2] + k.split("|")[3:]): v for k, v in deltas_by_obj[b].items()}
        common = set(ma) & set(mb)
        xs = [ma[k] for k in common]
        ys = [mb[k] for k in common]
        corr[f"{a}|{b}"] = {"n": len(common), "pearson": pearson(xs, ys)}

    # cycles on reliable edges only
    cycles = {}
    for obj, rs in by_obj.items():
        reliable = defaultdict(dict)
        allp = defaultdict(dict)
        for r in rs:
            if not r.get("valid") or r.get("pool") != "reference":
                continue
            key = r["prompt_id"]
            a, b = r["learner_id"], r["comparator_id"]
            allp[key][(a, b)] = 0.5 + r["delta"]
            allp[key][(b, a)] = 0.5 - r["delta"]
            ok = (r["mean_tie_probability"] < CONFIDENT_TIE_MASS
                  and abs(r["delta"]) > DECISIVE_MARGIN
                  and not r.get("unresolved_uncertainty"))
            if ok:
                reliable[key][(a, b)] = 0.5 + r["delta"]
                reliable[key][(b, a)] = 0.5 - r["delta"]
        hard = trip = 0
        soft = softn = 0.0
        devs = []
        for key, edges in allp.items():
            nodes = sorted({n for e in edges for n in e})
            for x, y, z in itertools.permutations(nodes, 3):
                if (x, y) in edges and (y, z) in edges and (z, x) in edges:
                    soft += edges[(x, y)] * edges[(y, z)] * edges[(z, x)]
                    softn += 1
            e = [(a, b, p) for (a, b), p in edges.items() if a < b]
            dv = bt_deviance(e)
            if dv is not None:
                devs.append(dv)
        for key, edges in reliable.items():
            nodes = sorted({n for e in edges for n in e})
            for x, y, z in itertools.permutations(nodes, 3):
                if (x, y) in edges and (y, z) in edges and (z, x) in edges:
                    trip += 1
                    if edges[(x, y)] > 0.5 and edges[(y, z)] > 0.5 and edges[(z, x)] > 0.5:
                        hard += 1
        cycles[obj] = {"reliable_triples": trip,
                       "hard_cycle_rate_reliable_edges": hard / trip if trip else None,
                       "soft_cycle_mass_all_edges": soft / softn if softn else None,
                       "mean_bt_deviance": (sum(devs) / len(devs)) if devs else None}

    manifest = json.loads(args.manifest.read_text()) if args.manifest else {}
    controls = {}
    if args.control_calibration and args.control_calibration.exists():
        c = json.loads(args.control_calibration.read_text())
        # The calibration may legitimately have selected NOTHING -- that is what
        # it reports when no candidate passes both known-answer prerequisites.
        # Fall back to the protocol this holdout actually ran, and say so.
        sel = c.get("selected")
        if sel is None:
            ran = (manifest.get("protocol_name") or "").split("_")[0].upper()
            sel = ran if ran in c["runs"] else None
            controls_note = ("calibration selected no admissible candidate; control "
                             f"metrics shown are for {sel}, the protocol this holdout ran")
        else:
            controls_note = "calibration-selected protocol"
        if sel is None:
            po = {}
        else:
            po = c["runs"][sel]["per_objective"]
        controls = {o: {"deterministic_degradation_accuracy":
                            v.get("deterministic_degradation_accuracy"),
                        "identical_confident_tie_accuracy":
                            v["identical_confident_tie_accuracy"],
                        "natural_pseudo_label_agreement":
                            v.get("directional_accuracy_natural"),
                        "clear_control_directional_accuracy_pooled":
                            v["directional_accuracy_clear"]}
                    for o, v in po.items()}
        controls["_note"] = controls_note

    failures = []
    for obj, v in per_obj.items():
        if v["invalid_rate"] >= GATES["invalid_rate_max"]:
            failures.append(f"{obj}: invalid rate {v['invalid_rate']:.4%}")
        if abs(v["signed_position_bias"]) > GATES["abs_signed_position_bias_max"]:
            failures.append(f"{obj}: |position bias| {abs(v['signed_position_bias']):.3f}")
        s = v["confident_decisive_swap_consistency"]
        if s is None or s < GATES["confident_decisive_swap_consistency_min"]:
            failures.append(f"{obj}: confident-decisive swap consistency "
                            f"{'n/a' if s is None else f'{s:.3f}'}")
        sh = v["split_half_template_spearman"] or v["split_half_order_spearman"]
        if sh is None or sh < GATES["split_half_spearman_min"]:
            failures.append(f"{obj}: split-half Spearman "
                            f"{'n/a' if sh is None else f'{sh:.3f}'}")
        if v["unresolved_uncertainty_rate"] > GATES["unresolved_uncertainty_rate_max"]:
            failures.append(f"{obj}: unresolved uncertainty "
                            f"{v['unresolved_uncertainty_rate']:.3f}")
        # Amendment 001: the deterministic and identical controls are known-answer
        # and gate; the UltraFeedback natural agreement is a GPT-4-derived
        # pseudo-label and is reported as a diagnostic only.
        c = controls.get(obj, {}) if isinstance(controls.get(obj), dict) else {}
        d = c.get("deterministic_degradation_accuracy")
        if d is not None and d < 0.90:
            failures.append(f"{obj}: deterministic degradation accuracy {d:.3f} "
                            "(development controls)")
        t_ = c.get("identical_confident_tie_accuracy")
        if t_ is not None and t_ < 0.90:
            failures.append(f"{obj}: identical-pair confident-tie accuracy {t_:.3f} "
                            "(development controls)")

    report = {"gates": GATES, "per_objective": per_obj,
              "objective_label_correlation": corr, "cycles": cycles,
              "control_based_gates": controls,
              "control_gate_caveat": ("the deterministic and identical-pair gates are "
                                      "evaluated on the DEVELOPMENT controls, which also "
                                      "informed protocol selection, so they are not "
                                      "independent evidence. Per Amendment 001 the "
                                      "UltraFeedback natural agreement is reported as a "
                                      "diagnostic and does not gate."),
              "run_manifest": manifest,
              "total_judge_calls": manifest.get("n_renderings"),
              "wall_clock_seconds": manifest.get("wall_clock_seconds"),
              "gate_failures": failures, "gates_passed": not failures}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))

    L = ["# Section F — frozen-protocol holdout audit", "",
         f"Protocol: `{manifest.get('protocol_name')}`  ",
         f"Pairs: **{sum(v['n_pairs'] for v in per_obj.values())}**, judge calls "
         f"**{manifest.get('n_renderings')}**, wall clock "
         f"{manifest.get('wall_clock_seconds')} s", "",
         f"## Gates: **{'PASS' if not failures else 'FAIL'}**", ""]
    for f in failures:
        L.append(f"- FAIL: {f}")
    if not failures:
        L.append("- every gate met")
    L += ["", "## Per objective", "",
          "| objective | pairs | invalid | adjud. | unresolved | raw swap | **confident swap** "
          "| contradiction | position bias (95% CI) | length bias | mean \\|D\\| | conf. tie |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for obj, v in per_obj.items():
        f = lambda x, d=3: "n/a" if x is None else f"{x:.{d}f}"
        ci = v["position_bias_ci95"]
        L.append(f"| {obj} | {v['n_pairs']} | {v['invalid_rate']:.3%} | "
                 f"{v['adjudicated_rate']:.3f} | {v['unresolved_uncertainty_rate']:.3f} | "
                 f"{f(v['raw_all_pair_hard_swap_consistency'])} | "
                 f"**{f(v['confident_decisive_swap_consistency'])}** | "
                 f"{f(v['decisive_contradiction_rate'])} | "
                 f"{v['signed_position_bias']:+.4f} "
                 f"[{f(ci[0],4)}, {f(ci[1],4)}] | {f(v['response_length_bias'])} | "
                 f"{v['mean_abs_delta']:.4f} | {v['confident_tie_rate']:.3f} |")
    L += ["", "## Reliability", "",
          "| objective | split-half (order) rho | split-half (template) rho | n template pairs |",
          "|---|---|---|---|"]
    for obj, v in per_obj.items():
        f = lambda x: "n/a" if x is None else f"{x:.3f}"
        L.append(f"| {obj} | {f(v['split_half_order_spearman'])} | "
                 f"{f(v['split_half_template_spearman'])} | "
                 f"{v['n_split_half_template_pairs']} |")
    L += ["", "## Cycles and transitivity", "",
          "| objective | reliable triples | hard cycle rate | soft cycle mass | BT deviance |",
          "|---|---|---|---|---|"]
    for obj, c in cycles.items():
        f = lambda x: "n/a" if x is None else f"{x:.4f}"
        L.append(f"| {obj} | {c['reliable_triples']} | "
                 f"{f(c['hard_cycle_rate_reliable_edges'])} | "
                 f"{f(c['soft_cycle_mass_all_edges'])} | {f(c['mean_bt_deviance'])} |")
    L += ["", "## Objective-label correlation", "", "| pair | n | pearson |", "|---|---|---|"]
    for k, v in corr.items():
        p = v["pearson"]
        L.append(f"| {k.replace('|',' vs ')} | {v['n']} | "
                 f"{'n/a' if p is None else f'{p:+.3f}'} |")
    (args.out_dir / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
