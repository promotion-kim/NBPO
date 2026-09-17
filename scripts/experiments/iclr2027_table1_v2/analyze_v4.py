#!/usr/bin/env python3
"""v4 protocol metrics and lexicographic selection.

Combines two evidence sources per protocol -- known-answer controls and real
same-policy pairs -- into the per-objective table the governing instruction
asks for, and applies the selection order:

  1 final unresolved invalid rate      5 order split-half Delta reliability
  2 deterministic-degradation accuracy 6 template split-half Delta reliability
  3 identical-pair confident-tie       7 max |position bias|
  4 confident semantic swap            8 judge-call and token cost

Eligibility is checked *before* ranking: a protocol failing final-unresolved
invalid < 0.2%, deterministic >= 0.90, or identical >= 0.90 on any objective is
not a candidate at whatever it scores elsewhere. When nothing is eligible the
tool says so rather than naming the least-bad, which is how an inadmissible
protocol got frozen in v3.

UltraFeedback natural agreement is computed and reported but never gates, per
Amendment 001.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

FORWARD, REVERSE = "learner_first", "comparator_first"
DECISIVE, CONF_TIE_MASS, CONF_TIE_DELTA = 0.1, 0.5, 0.05


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def pearson(x, y):
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    return num / (dx * dy) if dx > 0 and dy > 0 else None


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(o):
            r[i] = pos
        return r
    return pearson(rank(x), rank(y))


def half(obs, mode, which):
    if mode == "order":
        sel = [o for o in obs if o["presentation_order"] ==
               (FORWARD if which == "a" else REVERSE)]
    else:
        tids = sorted({o["template_id"] for o in obs})
        if len(tids) < 2:
            return None
        keep = {tids[0]} if which == "a" else set(tids[1:])
        sel = [o for o in obs if o["template_id"] in keep]
    if not sel:
        return None
    return sum(o["semantic_score"] for o in sel) / len(sel) - 0.5


def positions(rows):
    v = [o.get("verdict_token_position") for r in rows if r.get("valid")
         for o in (r.get("observations") or []) if o.get("verdict_token_position")]
    if not v:
        return {}
    v.sort()
    q = lambda f: v[min(len(v) - 1, int(f * len(v)))]
    return {"p50": q(.50), "p90": q(.90), "p95": q(.95), "p99": q(.99), "max": v[-1]}


def analyse(dev_rows, ctl_rows):
    """Per-objective metrics from real pairs (dev_rows) and controls (ctl_rows)."""
    all_dev, val_dev = defaultdict(list), defaultdict(list)
    for r in dev_rows:
        all_dev[r["objective"]].append(r)
        if r.get("valid"):
            val_dev[r["objective"]].append(r)
    ctl = defaultdict(list)
    for r in ctl_rows:
        ctl[r["objective"]].append(r)

    out = {}
    for obj in sorted(all_dev):
        rs, allr = val_dev[obj], all_dev[obj]
        c = [r for r in ctl[obj] if r.get("valid")]
        c_all = ctl[obj]

        def direction(fam):
            sub = [r for r in c if r.get("family") == fam]
            if not sub:
                return None
            ok = sum(1 for r in sub
                     if (r["expected_verdict"] == "A" and r["delta"] > 0)
                     or (r["expected_verdict"] == "B" and r["delta"] < 0))
            return ok / len(sub)

        ident = [r for r in c if r.get("family") == "identical"]
        ident_acc = (sum(1 for r in ident
                         if r["mean_tie_probability"] >= CONF_TIE_MASS
                         and abs(r["delta"]) <= CONF_TIE_DELTA) / len(ident)
                     if ident else None)

        both = [r for r in rs if abs(r["semantic_score_forward"] - 0.5) > DECISIVE
                and abs(r["semantic_score_reverse"] - 0.5) > DECISIVE]
        conf = [r for r in both if r["mean_tie_probability"] < CONF_TIE_MASS]
        agree = lambda S: sum(1 for r in S if (r["semantic_score_forward"] - 0.5)
                              * (r["semantic_score_reverse"] - 0.5) > 0)
        bias = [r["semantic_score_forward"] - r["semantic_score_reverse"] for r in rs]

        halves = {}
        for mode in ("order", "template"):
            a, b = [], []
            for r in rs:
                x = half(r.get("observations") or [], mode, "a")
                y = half(r.get("observations") or [], mode, "b")
                if x is not None and y is not None:
                    a.append(x)
                    b.append(y)
            halves[mode] = {"pearson": pearson(a, b), "spearman": spearman(a, b), "n": len(a)}

        d = [r["delta"] for r in rs]
        out[obj] = {
            "n_pairs_attempted": len(allr),
            "n_pairs_valid": len(rs),
            "final_unresolved_invalid_rate": sum(1 for r in allr if not r.get("valid")) / max(1, len(allr)),
            "first_pass_invalid_rate": sum(
                1 for r in allr for o in (r.get("observations") or [])
                if o.get("first_pass_invalid")) / max(1, len(allr)),
            "control_final_unresolved_invalid_rate": sum(
                1 for r in c_all if not r.get("valid")) / max(1, len(c_all)),
            "deterministic_degradation_accuracy": direction("degradation"),
            "identical_confident_tie_accuracy": ident_acc,
            "uf_natural_pseudo_label_agreement": direction("natural"),
            "raw_all_pair_swap_consistency": (agree(both) / len(both)) if both else None,
            "confident_semantic_swap_consistency": (agree(conf) / len(conf)) if conf else None,
            "contradiction_rate": ((len(both) - agree(both)) / len(both)) if both else None,
            "signed_position_bias": sum(bias) / max(1, len(bias)),
            "response_length_bias": None,
            "tie_rate": sum(1 for r in rs if r["mean_tie_probability"] >= CONF_TIE_MASS
                            and abs(r["delta"]) <= CONF_TIE_DELTA) / max(1, len(rs)),
            "unresolved_uncertainty_rate": sum(
                1 for r in rs if r.get("unresolved_uncertainty")) / max(1, len(rs)),
            "order_split_half_pearson": halves["order"]["pearson"],
            "order_split_half_spearman": halves["order"]["spearman"],
            "template_split_half_pearson": halves["template"]["pearson"],
            "template_split_half_spearman": halves["template"]["spearman"],
            "n_template_split_pairs": halves["template"]["n"],
            "mean_abs_delta": sum(abs(x) for x in d) / max(1, len(d)),
            "verdict_token_position": positions(allr),
        }
    return out


def summarise(per_obj):
    g = lambda k: [v[k] for v in per_obj.values() if v[k] is not None]
    return {
        "max_final_unresolved_invalid_rate": max(
            v["final_unresolved_invalid_rate"] for v in per_obj.values()),
        "min_deterministic_degradation_accuracy": min(g("deterministic_degradation_accuracy"), default=None),
        "min_identical_confident_tie_accuracy": min(g("identical_confident_tie_accuracy"), default=None),
        "min_confident_semantic_swap_consistency": min(g("confident_semantic_swap_consistency"), default=None),
        "min_order_split_half_spearman": min(g("order_split_half_spearman"), default=None),
        "min_template_split_half_spearman": min(g("template_split_half_spearman"), default=None),
        "max_abs_position_bias": max(abs(v["signed_position_bias"]) for v in per_obj.values()),
    }


def eligible(s):
    return (s["max_final_unresolved_invalid_rate"] < 0.002
            and (s["min_deterministic_degradation_accuracy"] or 0) >= 0.90
            and (s["min_identical_confident_tie_accuracy"] or 0) >= 0.90)


def key(s, cost):
    g = lambda v, d: d if v is None else v
    return (-s["max_final_unresolved_invalid_rate"],
            g(s["min_deterministic_degradation_accuracy"], -1),
            g(s["min_identical_confident_tie_accuracy"], -1),
            g(s["min_confident_semantic_swap_consistency"], -1),
            g(s["min_order_split_half_spearman"], -1),
            g(s["min_template_split_half_spearman"], -1),
            -s["max_abs_position_bias"], -cost)


def sibling_manifest(results_path: Path, explicit: str | None = None) -> dict:
    """The run manifest that goes with a results file.

    `run_judge_protocol` writes `<tag>_results.jsonl` and `<tag>_run_manifest.json`
    side by side, so the manifest is derivable and does not have to be passed.
    """
    if explicit:
        p = Path(explicit)
        return json.loads(p.read_text()) if p.exists() else {}
    p = Path(str(results_path).replace("_results.jsonl", "_run_manifest.json"))
    return json.loads(p.read_text()) if p.exists() else {}


def assert_same_protocol(name, dev_path, dev_man, ctl_path, ctl_man) -> None:
    """Refuse to score a protocol against controls judged by a DIFFERENT protocol.

    This is not hypothetical. The v4 development directory holds two P2-long runs
    -- a 2-template cost-parity cut and the restored 3-template protocol -- whose
    control files are named alike and sit one directory apart. Pairing the
    3-template dev run with the 2-template controls silently moved truthfulness
    deterministic-degradation accuracy from 0.960 to 0.860, which is the
    difference between passing and failing a hard eligibility gate. Every metric
    here mixes the two sources, so a mismatch is never merely cosmetic.
    """
    a, b = dev_man.get("protocol_sha256"), ctl_man.get("protocol_sha256")
    if a and b and a != b:
        raise SystemExit(
            f"protocol {name!r}: dev results {dev_path} were produced by protocol "
            f"{a[:16]}... but the controls {ctl_path} by {b[:16]}.... Known-answer "
            "gates and real-pair metrics must come from the SAME frozen protocol; "
            "re-run one side or pass the matching file.")
    if not a or not b:
        print(f"  [warn] protocol {name!r}: no protocol_sha256 on "
              f"{'dev' if not a else 'controls'} manifest; the match is unverified",
              flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", action="append", required=True,
                    help="name=dev_results.jsonl,ctl_results.jsonl[,manifest.json]")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--label", default="development")
    args = ap.parse_args()

    runs = {}
    for spec in args.protocol:
        name, paths = spec.split("=", 1)
        parts = paths.split(",")
        dev_path, ctl_path = Path(parts[0]), Path(parts[1])
        dev, ctl = read_jsonl(dev_path), read_jsonl(ctl_path)
        dev_man = sibling_manifest(dev_path, parts[2] if len(parts) > 2 else None)
        ctl_man = sibling_manifest(ctl_path, parts[3] if len(parts) > 3 else None)
        assert_same_protocol(name, dev_path, dev_man, ctl_path, ctl_man)
        po = analyse(dev, ctl)
        runs[name] = {"per_objective": po, "summary": summarise(po),
                      "manifest": dev_man, "controls_manifest": ctl_man,
                      "protocol_sha256": dev_man.get("protocol_sha256"),
                      "dev_results": str(dev_path), "control_results": str(ctl_path),
                      "cost_renderings": dev_man.get("n_renderings"),
                      "wall_clock_seconds": dev_man.get("wall_clock_seconds")}

    ranked = sorted(runs.items(), key=lambda kv: key(kv[1]["summary"],
                                                     kv[1].get("cost_renderings") or 0),
                    reverse=True)
    elig = [k for k, v in ranked if eligible(v["summary"])]
    selected = elig[0] if elig else None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.label}_metrics.json").write_text(json.dumps(
        {"runs": runs, "ranking": [k for k, _ in ranked], "eligible": elig,
         "selected": selected,
         "eligibility_rule": ("final unresolved invalid < 0.2%, deterministic "
                              "degradation >= 0.90, identical confident-tie >= 0.90, "
                              "every objective"),
         "selection_inputs": ("calibration and reliability metrics only; no NBPO, "
                              "baseline or policy quantity consulted")}, indent=2))

    f = lambda v, d=3: "n/a" if v is None else f"{v:.{d}f}"
    L = [f"# v4 {args.label}", "", "## Ranking", ""]
    for i, (n, r) in enumerate(ranked, 1):
        s = r["summary"]
        L.append(f"{i}. **{n}** — unresolved invalid {f(s['max_final_unresolved_invalid_rate'],4)}"
                 f"{'' if s['max_final_unresolved_invalid_rate'] < 0.002 else ' **(FAIL)**'}"
                 f", deterministic {f(s['min_deterministic_degradation_accuracy'])}"
                 f", identical {f(s['min_identical_confident_tie_accuracy'])}"
                 f", confident swap {f(s['min_confident_semantic_swap_consistency'])}"
                 f", order rho {f(s['min_order_split_half_spearman'])}"
                 f", template rho {f(s['min_template_split_half_spearman'])}"
                 f", max |bias| {f(s['max_abs_position_bias'])}"
                 f", cost {r.get('cost_renderings')}")
    L += ["", (f"**Selected: `{selected}`**" if selected else
               "**NO ELIGIBLE PROTOCOL** — every candidate fails a known-answer "
               "prerequisite. The ranking is diagnostic only."), ""]
    for n, r in runs.items():
        L += [f"## {n}", "",
              "| objective | unresolved inv | 1st-pass inv | determ. | identical | UF nat "
              "| raw swap | conf swap | contra | bias | tie | unresolved unc | order rho "
              "| tmpl rho | tok p50/p90/p99/max |", "|" + "---|" * 15]
        for o, v in r["per_objective"].items():
            tp = v["verdict_token_position"]
            L.append("| " + " | ".join([
                o, f(v['final_unresolved_invalid_rate'], 4), f(v['first_pass_invalid_rate'], 4),
                f(v['deterministic_degradation_accuracy']), f(v['identical_confident_tie_accuracy']),
                f(v['uf_natural_pseudo_label_agreement']), f(v['raw_all_pair_swap_consistency']),
                f(v['confident_semantic_swap_consistency']), f(v['contradiction_rate']),
                f"{v['signed_position_bias']:+.4f}", f(v['tie_rate']),
                f(v['unresolved_uncertainty_rate']), f(v['order_split_half_spearman']),
                f(v['template_split_half_spearman']),
                f"{tp.get('p50','-')}/{tp.get('p90','-')}/{tp.get('p99','-')}/{tp.get('max','-')}"]) + " |")
        L.append("")
    (args.out_dir / f"{args.label}_report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
