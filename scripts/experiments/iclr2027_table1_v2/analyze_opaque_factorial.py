#!/usr/bin/env python3
"""Decompose the judge's slot bias into a physical and a lexical part, and decide.

The factorial crosses semantic assignment, physical slot and identifier
labelling, so two effects that the ordinary rendering confounds perfectly come
apart exactly:

``position effect``    mean advantage of whichever response is rendered FIRST,
                       averaged over both labellings -- what a judge that reads
                       the first block more carefully would produce;
``identifier effect``  mean advantage of whichever response carries the FIRST
                       identifier ("A", or "K7"), averaged over both slots --
                       what a judge that prefers the letter A would produce.

Both are reported per objective with a cluster bootstrap over pairs, for the
canonical A/B scheme and for the opaque scheme, so the question "would neutral
identifiers fix this" is answered by measurement.

The decision rule is applied exactly as pre-registered, all seven gates
conjunctive, and no gate is adjusted by anything found here. The two downstream
target gates are evaluated only if the five judge-level gates pass, because they
require a separate full-pool run; that ordering is a cost decision and is stated,
not a relaxation -- a failure among the first five already decides the outcome.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

OBJECTIVES = ("helpfulness", "honesty", "instruction_following", "truthfulness")
DECISIVE = 0.1

GATES = {
    "unresolved_invalid_rate": ("<", 0.002),
    "deterministic_degradation_accuracy": (">=", 0.90),
    "identical_confident_tie_accuracy": (">=", 0.90),
    "confident_semantic_swap_consistency": (">=", 0.85),
    "order_split_half_spearman": (">=", 0.75),
    "finite_pool_target_pearson": (">=", 0.90),
    "finite_pool_target_sign_agreement": (">=", 0.85),
}
JUDGE_LEVEL_GATES = list(GATES)[:5]


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    r = lambda v: np.argsort(np.argsort(np.asarray(v, float))).astype(float)
    return pearson(r(x), r(y))


def cell_means(obs, key):
    """Mean of ``key`` per (pair, cell) then per pair -- so every pair weighs once."""
    per_pair = defaultdict(list)
    for o in obs:
        per_pair[o["pair_id"]].append(o[key])
    return {k: float(np.mean(v)) for k, v in per_pair.items()}


def balanced_effect(obs, key, factor, level_a, level_b):
    """Effect of ``factor`` on ``key``, with the OTHER factors balanced by design.

    Because the design is a complete crossing, averaging within each level of the
    factor already balances everything else, so the difference of level means is
    the effect. The value is expressed as an advantage over 0.5.
    """
    a = [o[key] for o in obs if o[factor] == level_a]
    b = [o[key] for o in obs if o[factor] == level_b]
    if not a or not b:
        return None
    return float(np.mean(a) - 0.5), float(np.mean(b) - 0.5), len(a), len(b)


def bootstrap_effect(obs, key, n=2000, seed=0):
    """Cluster bootstrap over pairs for a single mean-advantage statistic."""
    rng = np.random.default_rng(seed)
    by_pair = defaultdict(list)
    for o in obs:
        by_pair[o["pair_id"]].append(o[key])
    keys = list(by_pair)
    if not keys:
        return None
    draws = []
    for _ in range(n):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        vals = [v for i in pick for v in by_pair[keys[i]]]
        draws.append(float(np.mean(vals) - 0.5))
    return {"estimate": float(np.mean([v for vs in by_pair.values() for v in vs]) - 0.5),
            "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "n_pairs": len(keys), "n_observations": sum(len(v) for v in by_pair.values())}


def swap_and_reliability(obs):
    """Semantic Δ per pair, split by physical position, with labels averaged out.

    A pair's forward estimate averages every rendering in which the learner is
    physically first -- across BOTH labellings and all templates -- so the
    identifier effect is removed from it by construction. That is what makes this
    swap-consistency number a test of the *position* confound specifically.
    """
    fwd, rev, by_tpl = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
    for o in obs:
        (fwd if o["position"] == "y_first" else rev)[o["pair_id"]].append(o["win_learner"])
        by_tpl[o["template_id"]][o["pair_id"]].append(o["win_learner"])
    pairs = sorted(set(fwd) & set(rev))
    f = np.array([np.mean(fwd[p]) for p in pairs])
    r = np.array([np.mean(rev[p]) for p in pairs])
    p_hat = 0.5 * (f + r)
    dec = (np.abs(f - 0.5) > DECISIVE) & (np.abs(r - 0.5) > DECISIVE)
    agree = ((f - 0.5) * (r - 0.5) > 0)
    tpl_ids = sorted(by_tpl)
    tpl = None
    if len(tpl_ids) >= 2:
        a = [np.mean(by_tpl[tpl_ids[0]][p]) - 0.5 for p in pairs]
        b = [np.mean([v for t in tpl_ids[1:] for v in by_tpl[t][p]]) - 0.5 for p in pairs]
        tpl = {"pearson": pearson(a, b), "spearman": spearman(a, b)}
    return {
        "n_pairs": len(pairs),
        "mean_abs_delta": float(np.abs(p_hat - 0.5).mean()),
        "raw_all_pair_swap_consistency": float(agree.mean()) if len(pairs) else None,
        "confident_semantic_swap_consistency": (float(agree[dec].mean())
                                                if dec.any() else None),
        "contradiction_rate": (float(1 - agree[dec].mean()) if dec.any() else None),
        "n_decisive_pairs": int(dec.sum()),
        "order_split_half_pearson": pearson(f - 0.5, r - 0.5),
        "order_split_half_spearman": spearman(f - 0.5, r - 0.5),
        "template_split_half": tpl,
    }


def control_metrics(obs):
    """Known-answer accuracy, expressed in the SAME semantic orientation.

    A control row's ``expected_verdict`` names the originally-better response in
    the source file's own A/B convention, where A is ``response_a`` -- which this
    runner always maps to the learner slot. So "A" means the learner should win.
    """
    out = {}
    for fam in ("degradation", "identical", "natural"):
        sub = [o for o in obs if o.get("family") == fam]
        if not sub:
            out[fam] = None
            continue
        per_pair = defaultdict(list)
        exp = {}
        for o in sub:
            per_pair[o["pair_id"]].append(o["win_learner"])
            exp[o["pair_id"]] = o.get("expected_verdict")
        if fam == "identical":
            ok = sum(1 for p, v in per_pair.items() if abs(np.mean(v) - 0.5) <= 0.05)
        else:
            ok = 0
            for p, v in per_pair.items():
                m = float(np.mean(v)) - 0.5
                if exp[p] == "A" and m > 0:
                    ok += 1
                elif exp[p] == "B" and m < 0:
                    ok += 1
        out[fam] = {"accuracy": ok / len(per_pair), "n": len(per_pair)}
    return out


def analyse_scheme(natural, controls, scheme, seed=0):
    nat = [o for o in natural if o["scheme"] == scheme and o.get("valid")]
    ctl = [o for o in controls if o["scheme"] == scheme and o.get("valid")]
    nat_all = [o for o in natural if o["scheme"] == scheme]
    ctl_all = [o for o in controls if o["scheme"] == scheme]
    out = {"n_renderings": len(nat_all) + len(ctl_all),
           "unresolved_invalid_rate": (
               sum(1 for o in nat_all + ctl_all if not o.get("valid"))
               / max(1, len(nat_all) + len(ctl_all))),
           "first_pass_invalid_rate": (
               sum(1 for o in nat_all + ctl_all if o.get("first_pass_invalid"))
               / max(1, len(nat_all) + len(ctl_all))),
           "per_objective": {}}
    for obj in OBJECTIVES:
        o_nat = [o for o in nat if o["objective"] == obj]
        o_ctl = [o for o in ctl if o["objective"] == obj]
        if not o_nat:
            continue
        pos = bootstrap_effect(o_nat, "win_slot1", seed=seed)
        lex = bootstrap_effect(o_nat, "win_first_id", seed=seed + 1)
        cm = control_metrics(o_ctl)
        tpl_means = {t: float(np.mean([o["win_learner"] for o in o_nat
                                       if o["template_id"] == t]))
                     for t in sorted({o["template_id"] for o in o_nat})}
        out["per_objective"][obj] = {
            "physical_first_position_effect": pos,
            "identifier_effect": lex,
            "template_means_win_learner": tpl_means,
            "template_effect_range": (max(tpl_means.values()) - min(tpl_means.values())
                                      if tpl_means else None),
            **swap_and_reliability(o_nat),
            "deterministic_degradation_accuracy": (cm["degradation"] or {}).get("accuracy"),
            "identical_confident_tie_accuracy": (cm["identical"] or {}).get("accuracy"),
            "uf_natural_pseudo_label_agreement": (cm["natural"] or {}).get("accuracy"),
            "unresolved_invalid_rate": (
                sum(1 for o in nat_all + ctl_all
                    if o["objective"] == obj and not o.get("valid"))
                / max(1, sum(1 for o in nat_all + ctl_all if o["objective"] == obj))),
            "finite_pool_target_pearson": None,
            "finite_pool_target_sign_agreement": None,
        }
    return out


def decide(scheme_block):
    """The pre-registered rule, applied without adjustment."""
    checks, worst = [], {}
    for gate, (op, thr) in GATES.items():
        vals = {o: v.get(gate) for o, v in scheme_block["per_objective"].items()}
        present = {o: x for o, x in vals.items() if x is not None}
        if not present:
            checks.append({"gate": gate, "threshold": thr, "op": op,
                           "status": "NOT EVALUATED", "per_objective": vals})
            continue
        ok = all((x < thr) if op == "<" else (x >= thr) for x in present.values())
        checks.append({"gate": gate, "threshold": thr, "op": op,
                       "status": "PASS" if ok else "FAIL", "per_objective": present})
        worst[gate] = (max(present.values()) if op == "<" else min(present.values()))
    judge_level = [c for c in checks if c["gate"] in JUDGE_LEVEL_GATES]
    judge_pass = all(c["status"] == "PASS" for c in judge_level)
    unevaluated = [c["gate"] for c in checks if c["status"] == "NOT EVALUATED"]
    return {"checks": checks, "worst_per_gate": worst,
            "judge_level_gates_pass": judge_pass,
            "gates_not_evaluated": unevaluated,
            "decision": ("A: freeze one v5 candidate and run exactly one fresh holdout"
                         if judge_pass and not unevaluated else
                         "B: permanently retire prompted-Qwen judging from the main "
                         "experiment; no further prompt variants, no lowered thresholds")
            if not (judge_pass and unevaluated) else
            "A-pending: judge-level gates pass; the two downstream target gates must "
            "be run before freezing"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260908)
    args = ap.parse_args()

    natural = read_jsonl(args.run_dir / "natural_observations.jsonl")
    ctl_path = args.run_dir / "controls_observations.jsonl"
    controls = read_jsonl(ctl_path) if ctl_path.exists() else []
    manifest = json.loads((args.run_dir / "run_manifest.json").read_text())

    report = {"manifest": manifest, "schemes": {}}
    for scheme in sorted({o["scheme"] for o in natural}):
        blk = analyse_scheme(natural, controls, scheme, seed=args.seed)
        blk["decision"] = decide(blk)
        report["schemes"][scheme] = blk

    # the design check the instruction asks for, verified from the written file
    bal = defaultdict(lambda: defaultdict(int))
    for o in natural:
        bal[o["scheme"]][(o["position"], o["label_order"])] += 1
    report["design_balance"] = {s: {f"{k[0]}|{k[1]}": v for k, v in d.items()}
                               for s, d in bal.items()}
    report["design_is_balanced"] = all(len(set(d.values())) == 1 for d in bal.values())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "opaque_factorial_metrics.json").write_text(
        json.dumps(report, indent=2, default=str))

    L = ["# Opaque-identifier factorial", "",
         f"balanced design verified from file: **{report['design_is_balanced']}**", ""]
    for scheme, blk in report["schemes"].items():
        L += [f"## scheme `{scheme}`", "",
              f"unresolved invalid {blk['unresolved_invalid_rate']:.4f}  "
              f"(first pass {blk['first_pass_invalid_rate']:.4f})", "",
              "| objective | position effect [95% CI] | identifier effect [95% CI] | "
              "tmpl range | conf swap | contra | order rho | determ | identical |",
              "|" + "---|" * 9]
        for obj, v in blk["per_objective"].items():
            f = lambda x, d=3: "n/a" if x is None else f"{x:.{d}f}"
            pe, le = v["physical_first_position_effect"], v["identifier_effect"]
            ci = lambda e: (f"{e['estimate']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]"
                            if e else "n/a")
            L.append("| " + " | ".join([
                obj, ci(pe), ci(le), f(v["template_effect_range"]),
                f(v["confident_semantic_swap_consistency"]), f(v["contradiction_rate"]),
                f(v["order_split_half_spearman"]),
                f(v["deterministic_degradation_accuracy"]),
                f(v["identical_confident_tie_accuracy"])]) + " |")
        L += ["", "### gates", ""]
        for c in blk["decision"]["checks"]:
            L.append(f"- **{c['status']}**  `{c['gate']}` {c['op']} {c['threshold']}"
                     + ("" if c["status"] == "NOT EVALUATED"
                        else "  -> " + ", ".join(f"{k}={v:.3f}"
                                                 for k, v in c["per_objective"].items())))
        L += ["", f"**Decision: {blk['decision']['decision']}**", ""]
    (args.out_dir / "opaque_factorial_report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
