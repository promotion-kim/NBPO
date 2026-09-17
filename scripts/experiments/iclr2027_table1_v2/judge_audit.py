#!/usr/bin/env python3
"""Audit a judgment bank before spending a full labelling run on it.

The expensive mistake is discovering after 500k judge calls that the template
was applied wrong, that thinking mode leaked reasoning into the verdict slot, or
that the judge is really just picking whichever response is shown first. This
reads a bank produced by ``scripts.nbpo.judge_pairwise_matrix`` and reports, per
objective, the quantities that would reveal each of those.

Hard gates (Section 8 of the protocol) -- these are *not* tunable, and a failing
gate is reported as a failure rather than met by moving the threshold:

* invalid parse rate < 0.2 %
* no missing response or judgment cells
* swap-consistent winner rate >= 85 % after accounting for ties

Definitions, because they are easy to state loosely:

**Swap consistency.** ``policy_win`` is already expressed from the learner's
point of view in both presentation orders, so a judge with no position bias
gives the same value in both. The gate quantity is measured over the pairs where
**both orders returned a decisive verdict** -- that is what "after accounting for
ties" means. Two other readings are reported beside it and must not be confused
with it:

* a pair that is a tie in one order and a win in the other is *partial*
  agreement, not a contradiction; counting it as one drags the rate down by 20
  points or more on a pool this close, so it is reported separately as
  ``one_order_tie_rate``;
* ``exact_agreement_rate`` counts both-tie agreement as agreement, which
  flatters a judge that ties everything.

``contradiction_rate`` -- both orders decisive and pointing opposite ways -- is
the number that actually indicts a judge, and it is reported explicitly.

**Position bias.** ``mean(policy_win | learner shown first) - mean(policy_win |
learner shown second)``. A judge that always picks slot A scores +1 in the first
and 0 in the second, so this is +1; an unbiased judge scores 0. This is exactly
the quantity swap averaging cancels, which is why the bank stores both orders.

**Cycles.** Over the swap-averaged margins on one prompt's response pool: a
directed triangle ``i > j > k > i`` where every edge exceeds the predeclared
margin threshold 0.1. The threshold is declared here, before any policy is
trained, so it cannot be chosen to make a number look better later.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

GATES = {
    "invalid_parse_rate_max": 0.002,
    "swap_consistent_winner_rate_min": 0.85,
    "cycle_margin_threshold": 0.1,
}


def load_rows(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def semantic_key(r):
    return (r["prompt_id"], r["learner_pool"], r["learner_response_id"],
            r["comparator_response_id"])


def audit(rows, response_lengths=None):
    by_obj = defaultdict(list)
    for r in rows:
        by_obj[r["objective"]].append(r)

    report = {"gates": GATES, "objectives": {}, "totals": {}}
    total_calls = len(rows)
    total_invalid = sum(1 for r in rows if not r.get("valid", False))
    margins_by_obj = {}

    for obj, obj_rows in sorted(by_obj.items()):
        pairs = defaultdict(dict)
        for r in obj_rows:
            pairs[semantic_key(r)][r["presentation_order"]] = r

        invalid = sum(1 for r in obj_rows if not r.get("valid", False))
        retries = sum(1 for r in obj_rows if int(r.get("attempt", 0)) > 0)
        first_vals, second_vals = [], []
        agree = contradict = both_tie = one_tie = exact_equal = 0
        margins, tie_hits, decided = {}, 0, 0
        missing_order = 0
        len_diffs, win_signs = [], []

        for key, orders in pairs.items():
            a = orders.get("learner_first")
            b = orders.get("comparator_first")
            if a is None or b is None:
                missing_order += 1
                continue
            qa, qb = float(a["policy_win"]), float(b["policy_win"])
            first_vals.append(qa)
            second_vals.append(qb)
            # swap-averaged probability that the learner beats the comparator
            p_hat = 0.5 * (qa + qb)
            margins[key] = p_hat - 0.5
            if abs(p_hat - 0.5) < 1e-12:
                tie_hits += 1
            else:
                decided += 1
            exact_equal += int(qa == qb)
            if qa != 0.5 and qb != 0.5:
                if qa == qb:
                    agree += 1
                else:
                    contradict += 1
            elif qa == 0.5 and qb == 0.5:
                both_tie += 1
            else:
                one_tie += 1
            if response_lengths:
                la = response_lengths.get(a["learner_response_id"], {}).get(key[0])
                lb = response_lengths.get(a["comparator_response_id"], {}).get(key[0])
                if la is not None and lb is not None:
                    len_diffs.append(la - lb)
                    win_signs.append(p_hat - 0.5)

        n_pairs = agree + contradict + both_tie + one_tie
        both_decisive = agree + contradict
        margins_by_obj[obj] = margins

        report["objectives"][obj] = {
            "total_calls": len(obj_rows),
            "semantic_pairs": len(pairs),
            "pairs_missing_an_order": missing_order,
            "invalid_parse_rate": invalid / max(1, len(obj_rows)),
            "retry_rate": retries / max(1, len(obj_rows)),
            "tie_rate_swap_averaged": tie_hits / max(1, tie_hits + decided),
            "both_orders_tie_rate": both_tie / max(1, n_pairs),
            "one_order_tie_rate": one_tie / max(1, n_pairs),
            "both_orders_decisive": both_decisive,
            # THE GATE QUANTITY: of the pairs both orders decided, how many agree.
            "swap_consistent_winner_rate": (agree / both_decisive
                                            if both_decisive else None),
            "contradiction_rate": (contradict / both_decisive
                                   if both_decisive else None),
            "exact_agreement_rate": exact_equal / max(1, n_pairs),
            "position_bias": (sum(first_vals) / max(1, len(first_vals))
                              - sum(second_vals) / max(1, len(second_vals))),
            "mean_policy_win_learner_first": sum(first_vals) / max(1, len(first_vals)),
            "mean_policy_win_learner_second": sum(second_vals) / max(1, len(second_vals)),
            "length_bias_correlation": _pearson(len_diffs, win_signs) if len_diffs else None,
        }

    # --- objective-label correlation, over the shared semantic pairs ---------
    objs = sorted(margins_by_obj)
    corr = {}
    for a, b in itertools.combinations(objs, 2):
        shared = set(margins_by_obj[a]) & set(margins_by_obj[b])
        xs = [margins_by_obj[a][k] for k in shared]
        ys = [margins_by_obj[b][k] for k in shared]
        corr[f"{a}|{b}"] = {"n": len(shared), "pearson": _pearson(xs, ys)}
    report["objective_label_correlation"] = corr

    # --- cycles -------------------------------------------------------------
    report["cycles"] = cycle_report(margins_by_obj, GATES["cycle_margin_threshold"])

    report["totals"] = {
        "total_calls": total_calls,
        "invalid_parse_rate": total_invalid / max(1, total_calls),
        "objectives": objs,
    }

    # --- gate verdicts ------------------------------------------------------
    failures = []
    if report["totals"]["invalid_parse_rate"] >= GATES["invalid_parse_rate_max"]:
        failures.append(f"invalid parse rate {report['totals']['invalid_parse_rate']:.4%} "
                        f">= {GATES['invalid_parse_rate_max']:.2%}")
    for obj, o in report["objectives"].items():
        if o["pairs_missing_an_order"]:
            failures.append(f"{obj}: {o['pairs_missing_an_order']} semantic pairs judged in "
                            "only one presentation order")
        rate = o["swap_consistent_winner_rate"]
        if rate is None:
            failures.append(f"{obj}: no decidable pairs, swap consistency undefined")
        elif rate < GATES["swap_consistent_winner_rate_min"]:
            failures.append(f"{obj}: swap-consistent winner rate {rate:.3f} < "
                            f"{GATES['swap_consistent_winner_rate_min']}")
    report["gate_failures"] = failures
    report["gates_passed"] = not failures
    return report


def cycle_report(margins_by_obj, threshold):
    """Directed triangles among a prompt's responses, per objective."""
    out = {}
    for obj, margins in margins_by_obj.items():
        # margin[(prompt, pool, i, j)] = P(i > j) - 1/2, so the reverse edge is -m
        by_prompt = defaultdict(dict)
        for (pid, pool, i, j), m in margins.items():
            by_prompt[(pid, pool)][(i, j)] = m
            by_prompt[(pid, pool)][(j, i)] = -m
        hard = soft = triples = 0
        for edges in by_prompt.values():
            nodes = sorted({n for e in edges for n in e})
            for a, b, c in itertools.combinations(nodes, 3):
                for x, y, z in ((a, b, c), (a, c, b)):
                    e1, e2, e3 = edges.get((x, y)), edges.get((y, z)), edges.get((z, x))
                    if e1 is None or e2 is None or e3 is None:
                        continue
                    triples += 1
                    if e1 > threshold and e2 > threshold and e3 > threshold:
                        hard += 1
                    # soft mass: probability all three edges point the same way round
                    soft += ((e1 + 0.5) * (e2 + 0.5) * (e3 + 0.5))
        out[obj] = {
            "directed_triples_examined": triples,
            "hard_directed_cycle_rate": hard / max(1, triples),
            "soft_cycle_mass": soft / max(1, triples),
            "margin_threshold": threshold,
        }
    return out


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else None


def render_markdown(report, source) -> str:
    L = [f"# Judge audit\n", f"Source: `{source}`\n",
         f"Total calls: **{report['totals']['total_calls']}**  ",
         f"Invalid parse rate: **{report['totals']['invalid_parse_rate']:.4%}** "
         f"(gate < {GATES['invalid_parse_rate_max']:.2%})\n",
         "## Gates\n",
         f"**{'PASS' if report['gates_passed'] else 'FAIL'}**\n"]
    for f in report["gate_failures"]:
        L.append(f"- FAIL: {f}")
    if not report["gate_failures"]:
        L.append("- every gate met; no threshold was moved to get here")
    L.append("\n## Per objective\n")
    L.append("| objective | calls | pairs | invalid | retry | tie rate | both-order "
             "decisive | swap-consistent (gate) | contradiction | position bias | "
             "length bias |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for obj, o in report["objectives"].items():
        sc = o["swap_consistent_winner_rate"]
        cr = o["contradiction_rate"]
        lb = o["length_bias_correlation"]
        L.append(f"| {obj} | {o['total_calls']} | {o['semantic_pairs']} | "
                 f"{o['invalid_parse_rate']:.3%} | {o['retry_rate']:.3%} | "
                 f"{o['tie_rate_swap_averaged']:.3f} | {o['both_orders_decisive']} | "
                 f"{'n/a' if sc is None else f'{sc:.3f}'} | "
                 f"{'n/a' if cr is None else f'{cr:.3f}'} | {o['position_bias']:+.4f} | "
                 f"{'n/a' if lb is None else f'{lb:+.3f}'} |")
    L.append("\n## Objective-label correlation (swap-averaged margins)\n")
    L.append("| pair | n | pearson |")
    L.append("|---|---|---|")
    for k, v in report["objective_label_correlation"].items():
        p = v["pearson"]
        L.append(f"| {k.replace('|', ' vs ')} | {v['n']} | "
                 f"{'n/a' if p is None else f'{p:+.3f}'} |")
    L.append("\n## Cycles\n")
    L.append(f"Margin threshold **{GATES['cycle_margin_threshold']}**, declared before any "
             "policy was trained.\n")
    L.append("| objective | triples | hard cycle rate | soft cycle mass |")
    L.append("|---|---|---|---|")
    for obj, c in report["cycles"].items():
        L.append(f"| {obj} | {c['directed_triples_examined']} | "
                 f"{c['hard_directed_cycle_rate']:.4f} | {c['soft_cycle_mass']:.4f} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdicts", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--responses", type=Path, nargs="*", default=None,
                    help="seed=path.json response files, for the length-bias column")
    args = ap.parse_args()

    lengths = None
    if args.responses:
        lengths = {}
        for spec in args.responses:
            seed, path = spec.split("=", 1)
            data = json.loads(Path(path).read_text())
            rows = data if isinstance(data, list) else data.get("responses", [])
            for r in rows:
                lengths.setdefault(seed, {})[r["prompt_id"]] = len(r.get("generated_text", ""))

    rows = load_rows(args.verdicts)
    report = audit(rows, lengths)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    (args.out_dir / "report.md").write_text(render_markdown(report, args.verdicts))
    print(render_markdown(report, args.verdicts))
    raise SystemExit(0 if report["gates_passed"] else 1)


if __name__ == "__main__":
    main()
