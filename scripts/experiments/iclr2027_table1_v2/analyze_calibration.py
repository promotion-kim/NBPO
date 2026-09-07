#!/usr/bin/env python3
"""Score judge protocols on the calibration controls, and select one.

Selection is lexicographic and uses **only** calibration and reliability
metrics. No NBPO surplus, no fixed-reference surplus, no Game-KS result, no
policy win rate, and no consideration of which protocol flatters the method
enters this file at all -- the ordering below is the whole decision rule:

1. maximize the minimum objective-wise directional accuracy on clear controls
2. maximize identical-pair confident-tie accuracy
3. minimize the maximum absolute position bias across objectives
4. maximize confident-pair semantic swap consistency
5. maximize split-half Delta reliability
6. break remaining ties by cost

Definitions that would otherwise be ambiguous:

**directional accuracy** -- on natural and degradation controls, whose expected
winner is known: does ``sign(Delta)`` point at the expected response? Ties count
as errors here, because these pairs are the ones with a real difference.

**confident tie** -- protocol-neutral, so P0 and P1 are judged the same way:
tie mass at least 0.5 *and* ``|Delta| <= 0.05``. A P0 pair that returns A in one
order and A in the other has ``Delta = 0`` but zero tie mass, so it is correctly
not counted as a confident tie -- that is position bias, not agreement.

**position bias** -- ``mean(s_forward - s_reverse)``. A judge that always picks
the shown-first response scores +1; an unbiased one scores 0.

**confident-pair swap consistency** -- among pairs where both orders commit
(``|s - 0.5| > 0.1``), the fraction whose direction agrees.

**split-half reliability** -- observations are split by template into two
halves, ``Delta`` is recomputed independently on each, and the two are
correlated. This needs every template scored on every pair (``--all-templates``),
so it is reported as ``None`` for a run that used the adaptive path.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

FORWARD, REVERSE = "learner_first", "comparator_first"
CHOICES = ("A", "B", "TIE")
CLEAR_FAMILIES = ("natural", "degradation")
CONFIDENT_TIE_MASS = 0.5
CONFIDENT_TIE_DELTA = 0.05
DECISIVE_MARGIN = 0.1


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
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(rank(xs), rank(ys))


def half_delta(observations, template_ids):
    """Delta recomputed from only the observations whose template is in the set."""
    fwd = [o for o in observations
           if o["template_id"] in template_ids and o["presentation_order"] == FORWARD]
    rev = [o for o in observations
           if o["template_id"] in template_ids and o["presentation_order"] == REVERSE]
    if not fwd or not rev:
        return None
    s_f = sum(o["semantic_score"] for o in fwd) / len(fwd)
    s_r = sum(o["semantic_score"] for o in rev) / len(rev)
    return 0.5 * (s_f + s_r) - 0.5


def analyse(rows):
    by_obj = defaultdict(list)
    for r in rows:
        if r.get("valid"):
            by_obj[r["objective"]].append(r)

    per_obj = {}
    for obj, rs in sorted(by_obj.items()):
        clear = [r for r in rs if r.get("family") in CLEAR_FAMILIES]
        ident = [r for r in rs if r.get("family") == "identical"]

        def directional(subset):
            if not subset:
                return None
            ok = 0
            for r in subset:
                want = r["expected_verdict"]
                d = r["delta"]
                ok += int((want == "A" and d > 0) or (want == "B" and d < 0))
            return ok / len(subset)

        conf_tie = None
        if ident:
            conf_tie = sum(
                1 for r in ident
                if r["mean_tie_probability"] >= CONFIDENT_TIE_MASS
                and abs(r["delta"]) <= CONFIDENT_TIE_DELTA) / len(ident)

        bias = (sum(r["semantic_score_forward"] - r["semantic_score_reverse"] for r in rs)
                / max(1, len(rs)))
        ident_bias = (sum(r["semantic_score_forward"] - r["semantic_score_reverse"]
                          for r in ident) / len(ident)) if ident else None

        both_commit = [r for r in rs
                       if abs(r["semantic_score_forward"] - 0.5) > DECISIVE_MARGIN
                       and abs(r["semantic_score_reverse"] - 0.5) > DECISIVE_MARGIN]
        swap = None
        if both_commit:
            swap = sum(1 for r in both_commit
                       if (r["semantic_score_forward"] - 0.5)
                       * (r["semantic_score_reverse"] - 0.5) > 0) / len(both_commit)

        h1, h2 = [], []
        for r in rs:
            obs = r.get("observations") or []
            tids = sorted({o["template_id"] for o in obs})
            if len(tids) < 2:
                continue
            a = half_delta(obs, {tids[0]})
            b = half_delta(obs, set(tids[1:]))
            if a is not None and b is not None:
                h1.append(a)
                h2.append(b)

        deltas = [r["delta"] for r in rs]
        per_obj[obj] = {
            "n": len(rs),
            "n_clear": len(clear), "n_identical": len(ident),
            "directional_accuracy_clear": directional(clear),
            "directional_accuracy_natural": directional(
                [r for r in rs if r.get("family") == "natural"]),
            "directional_accuracy_degradation": directional(
                [r for r in rs if r.get("family") == "degradation"]),
            "identical_confident_tie_accuracy": conf_tie,
            "position_bias": bias,
            "position_bias_identical_only": ident_bias,
            "confident_pair_swap_consistency": swap,
            "n_both_orders_commit": len(both_commit),
            "split_half_pearson": pearson(h1, h2) if h1 else None,
            "split_half_spearman": spearman(h1, h2) if h1 else None,
            "n_split_half_pairs": len(h1),
            "mean_abs_delta": sum(abs(d) for d in deltas) / max(1, len(deltas)),
            "mean_tie_probability": sum(r["mean_tie_probability"] for r in rs) / max(1, len(rs)),
            "mean_normalized_entropy": sum(r["mean_normalized_entropy"] for r in rs) / max(1, len(rs)),
            "unresolved_uncertainty_rate": sum(1 for r in rs
                                               if r.get("unresolved_uncertainty")) / max(1, len(rs)),
            "adjudicated_rate": sum(1 for r in rs if r.get("adjudicated")) / max(1, len(rs)),
        }

    invalid = sum(1 for r in rows if not r.get("valid"))
    def worst(key):
        vals = [v[key] for v in per_obj.values() if v[key] is not None]
        return min(vals) if vals else None
    def worst_abs(key):
        vals = [abs(v[key]) for v in per_obj.values() if v[key] is not None]
        return max(vals) if vals else None

    return {
        "per_objective": per_obj,
        "summary": {
            "invalid_rate": invalid / max(1, len(rows)),
            "min_directional_accuracy_clear": worst("directional_accuracy_clear"),
            "min_identical_confident_tie_accuracy": worst("identical_confident_tie_accuracy"),
            "max_abs_position_bias": worst_abs("position_bias"),
            "min_confident_pair_swap_consistency": worst("confident_pair_swap_consistency"),
            "min_split_half_spearman": worst("split_half_spearman"),
            "min_split_half_pearson": worst("split_half_pearson"),
        },
    }


def selection_key(summary, cost):
    """Section E's lexicographic order, as a sort key (all maximized except bias/cost)."""
    def g(v, default):
        return default if v is None else v
    return (g(summary["min_directional_accuracy_clear"], -1.0),
            g(summary["min_identical_confident_tie_accuracy"], -1.0),
            -g(summary["max_abs_position_bias"], 1e9),
            g(summary["min_confident_pair_swap_consistency"], -1.0),
            g(summary["min_split_half_spearman"], -1.0),
            -cost)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True,
                    help="name=results.jsonl[:manifest.json]")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    runs = {}
    for spec in args.run:
        name, paths = spec.split("=", 1)
        res, _, man = paths.partition(":")
        rows = read_jsonl(Path(res))
        manifest = json.loads(Path(man).read_text()) if man and Path(man).exists() else {}
        a = analyse(rows)
        a["manifest"] = manifest
        a["cost_renderings"] = manifest.get("n_renderings")
        a["wall_clock_seconds"] = manifest.get("wall_clock_seconds")
        runs[name] = a

    ranked = sorted(runs.items(),
                    key=lambda kv: selection_key(kv[1]["summary"],
                                                 kv[1].get("cost_renderings") or 0),
                    reverse=True)
    selected = ranked[0][0]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "calibration_results.json").write_text(
        json.dumps({"runs": runs, "ranking": [k for k, _ in ranked],
                    "selected": selected,
                    "selection_rule": ("lexicographic: min clear-control directional "
                                       "accuracy > identical confident-tie accuracy > "
                                       "-max |position bias| > confident-pair swap "
                                       "consistency > split-half Spearman > -cost"),
                    "selection_inputs": ("calibration and reliability metrics only; no "
                                         "downstream policy or NBPO quantity was "
                                         "consulted")}, indent=2))

    # flat CSV
    cols = ["protocol", "objective", "n", "directional_accuracy_clear",
            "directional_accuracy_natural", "directional_accuracy_degradation",
            "identical_confident_tie_accuracy", "position_bias",
            "position_bias_identical_only", "confident_pair_swap_consistency",
            "split_half_pearson", "split_half_spearman", "mean_abs_delta",
            "mean_tie_probability", "mean_normalized_entropy",
            "unresolved_uncertainty_rate", "adjudicated_rate"]
    lines = [",".join(cols)]
    for name, a in runs.items():
        for obj, v in a["per_objective"].items():
            lines.append(",".join(
                [name, obj] + ["" if v.get(c) is None else f"{v[c]:.6f}"
                               if isinstance(v.get(c), float) else str(v.get(c))
                               for c in cols[2:]]))
    (args.out_dir / "calibration_results.csv").write_text("\n".join(lines) + "\n")

    L = ["# Judge protocol calibration", "",
         "Selection uses calibration and reliability metrics only. No NBPO, "
         "fixed-reference, Game-KS or policy quantity was consulted, and the ordering "
         "below was fixed before the numbers were seen.", "",
         "## Ranking", ""]
    for i, (name, a) in enumerate(ranked, 1):
        s = a["summary"]
        L.append(f"{i}. **{name}** — min clear-control accuracy "
                 f"{fmt(s['min_directional_accuracy_clear'])}, identical confident-tie "
                 f"{fmt(s['min_identical_confident_tie_accuracy'])}, max |position bias| "
                 f"{fmt(s['max_abs_position_bias'])}, confident swap "
                 f"{fmt(s['min_confident_pair_swap_consistency'])}, split-half rho "
                 f"{fmt(s['min_split_half_spearman'])}, cost "
                 f"{a.get('cost_renderings')} renderings")
    L += ["", f"**Selected: `{selected}`**", "", "## Per objective", ""]
    for name, a in runs.items():
        L += [f"### {name}", "",
              "| objective | clear acc | natural | degrade | ident. tie | pos. bias | "
              "confident swap | split-half rho | mean \\|D\\| | tie mass | entropy |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for obj, v in a["per_objective"].items():
            L.append("| " + " | ".join([
                obj, fmt(v["directional_accuracy_clear"]),
                fmt(v["directional_accuracy_natural"]),
                fmt(v["directional_accuracy_degradation"]),
                fmt(v["identical_confident_tie_accuracy"]),
                f"{v['position_bias']:+.4f}",
                fmt(v["confident_pair_swap_consistency"]),
                fmt(v["split_half_spearman"]),
                f"{v['mean_abs_delta']:.4f}",
                f"{v['mean_tie_probability']:.3f}",
                f"{v['mean_normalized_entropy']:.3f}"]) + " |")
        L.append("")
    (args.out_dir / "calibration_report.md").write_text("\n".join(L))
    print("\n".join(L))


def fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


if __name__ == "__main__":
    main()
