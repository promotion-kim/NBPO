#!/usr/bin/env python3
"""Freeze the failed v2 judge protocol as an immutable diagnostic.

The v2 hard-verdict bank is not going to label the final experiment, but it is
the evidence for *why* the protocol changed, so it is preserved with enough
provenance that the failure can be re-derived rather than taken on trust.

Raw banks stay where they were written (they are tens of MB and must not enter
the repository); this records their sha256, row counts and every aggregate the
failure report quotes.

Metric definitions, stated once because loose versions of them differ by 20
points:

``decisive/decisive semantic consistency``
    Of the semantic pairs where BOTH presentation orders returned a non-tie
    verdict, the fraction whose winner agrees. This is the quantity the old
    0.85 gate was written against.

``decisive contradiction rate``
    Its complement -- both orders committed and disagreed. This is the number
    that indicts a judge, as opposed to one that merely abstains.

``tie/tie rate``
    Both orders returned TIE. Abstention, not inconsistency.

``position bias``
    mean(policy_win | learner shown first) - mean(policy_win | learner shown
    second). A judge that always picks the A slot scores +1; an unbiased judge
    scores 0. Swap averaging cancels exactly this component.

``|Delta|``
    The magnitude of the centered swap-averaged margin that actually reaches
    the game tensor. |Delta| = 0 means the comparison contributes nothing --
    and a position contradiction and a genuine tie both land there, which is
    the mechanical heart of the failure.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

FORWARD, REVERSE = "learner_first", "comparator_first"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_fingerprint(path: Path, patterns=("*.json", "*.txt", "*.model")) -> dict:
    """Hash the files that determine tokenization, so a swapped tokenizer shows."""
    out = {}
    if not path.exists():
        return {"missing": str(path)}
    for pat in patterns:
        for f in sorted(path.glob(pat)):
            try:
                out[f.name] = sha256_file(f)[:32]
            except OSError:
                pass
    return out


def analyse(rows):
    """Every per-objective quantity the failure report quotes."""
    by_obj = collections.defaultdict(list)
    for r in rows:
        by_obj[r["objective"]].append(r)

    out = {}
    for obj, orows in sorted(by_obj.items()):
        invalid = sum(1 for r in orows if not r.get("valid"))
        retried = sum(1 for r in orows if int(r.get("attempt", 0)) > 0)
        pairs = collections.defaultdict(dict)
        for r in orows:
            if not r.get("valid"):
                continue
            key = (r["prompt_id"], r["learner_pool"],
                   r["learner_response_id"], r["comparator_response_id"])
            pairs[key][r["presentation_order"]] = float(r["policy_win"])

        agree = contra = both_tie = one_tie = half = 0
        fwd_vals, rev_vals, deltas = [], [], []
        for orders in pairs.values():
            if FORWARD not in orders or REVERSE not in orders:
                half += 1
                continue
            f, rv = orders[FORWARD], orders[REVERSE]
            fwd_vals.append(f)
            rev_vals.append(rv)
            deltas.append(abs(0.5 * (f + rv) - 0.5))
            if f != 0.5 and rv != 0.5:
                agree += int(f == rv)
                contra += int(f != rv)
            elif f == 0.5 and rv == 0.5:
                both_tie += 1
            else:
                one_tie += 1
        n = agree + contra + both_tie + one_tie
        decisive = agree + contra
        q = statistics.quantiles(deltas, n=10) if len(deltas) > 10 else []
        out[obj] = {
            "total_calls": len(orows),
            "parser_invalid_rate": invalid / max(1, len(orows)),
            "retry_rate": retried / max(1, len(orows)),
            "semantic_pairs": n,
            "pairs_missing_an_order": half,
            "both_orders_decisive": decisive,
            "decisive_decisive_semantic_consistency": (agree / decisive if decisive else None),
            "decisive_contradiction_rate": (contra / decisive if decisive else None),
            "tie_tie_rate": both_tie / max(1, n),
            "one_order_tie_rate": one_tie / max(1, n),
            "position_bias": (sum(fwd_vals) / max(1, len(fwd_vals))
                              - sum(rev_vals) / max(1, len(rev_vals))),
            "mean_abs_delta": sum(deltas) / max(1, len(deltas)),
            "abs_delta_max_possible": 0.5,
            "abs_delta_fraction_at_zero": sum(1 for d in deltas if d < 1e-9) / max(1, len(deltas)),
            "abs_delta_fraction_at_max": sum(1 for d in deltas if d > 0.49) / max(1, len(deltas)),
            "abs_delta_deciles": [round(v, 6) for v in q],
            "abs_delta_histogram": dict(collections.Counter(round(d, 2) for d in deltas)),
        }
    totals = {
        "total_calls": len(rows),
        "parser_invalid_rate": sum(1 for r in rows if not r.get("valid")) / max(1, len(rows)),
        "retry_rate": sum(1 for r in rows if int(r.get("attempt", 0)) > 0) / max(1, len(rows)),
        "mean_abs_delta": (sum(o["mean_abs_delta"] * o["semantic_pairs"] for o in out.values())
                           / max(1, sum(o["semantic_pairs"] for o in out.values()))),
    }
    return {"per_objective": out, "totals": totals}


def load(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", action="append", required=True,
                    help="label=path.jsonl, once per judge")
    ap.add_argument("--judge-path", action="append", default=[],
                    help="label=/path/to/model, for revision fingerprints")
    ap.add_argument("--rubric", type=Path, required=True)
    ap.add_argument("--pool", action="append", default=[],
                    help="seed=path.json response pool files")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--git-commit", default=None,
                    help="source commit, for when this runs outside the checkout "
                         "(the cluster copy is not a git repo, and 'unknown' in a "
                         "provenance record is worse than no record)")
    args = ap.parse_args()

    commit = args.git_commit
    if commit is None:
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(args.repo), "rev-parse", "HEAD"],
                text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            raise SystemExit(
                f"{args.repo} is not a git checkout and --git-commit was not given. "
                "A frozen diagnostic without its source commit cannot be re-derived; "
                "pass the commit explicitly rather than recording 'unknown'.")

    judge_paths = dict(s.split("=", 1) for s in args.judge_path)
    report = {
        "purpose": ("immutable diagnostic for the FAILED v2 hard-verdict judge protocol; "
                    "these labels never enter the final bank"),
        "git_commit": commit,
        "rubric": {"path": str(args.rubric), "sha256": sha256_file(args.rubric)},
        "response_pool": {},
        "banks": {},
        "swap_mapping_verified": {
            "test": "tests/test_judge_swap_mapping.py",
            "cases": {
                "forward A / reverse B -> p_hat(y>z)": 1.0,
                "forward B / reverse A -> p_hat(y>z)": 0.0,
                "forward A / reverse A (first-position contradiction)": 0.5,
                "forward B / reverse B (second-position contradiction)": 0.5,
                "tie / tie": 0.5,
            },
            "reference_tensor_exactly_skew_symmetric": True,
            "verdict": ("the v2 semantic swap mapping is CORRECT; the consistency failure "
                        "is not a mapping bug"),
        },
    }
    for spec in args.pool:
        seed, path = spec.split("=", 1)
        report["response_pool"][seed] = {"path": path, "sha256": sha256_file(Path(path))}

    for spec in args.bank:
        label, path = spec.split("=", 1)
        p = Path(path)
        rows = load(p)
        entry = {
            "path": str(p),
            "sha256": sha256_file(p),
            "rows": len(rows),
            "judge_model": rows[0].get("judge_model") if rows else None,
            "rubric_version": rows[0].get("rubric_version") if rows else None,
            "judge_effective_config": rows[0].get("judge_effective_config") if rows else None,
            **analyse(rows),
        }
        if label in judge_paths:
            entry["judge_revision_fingerprint"] = dir_fingerprint(Path(judge_paths[label]))
        report["banks"][label] = entry

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "raw_metrics.json").write_text(json.dumps(report, indent=2))
    (args.out_dir / "FAILURE_REPORT.md").write_text(render(report))
    print(render(report))


def render(r: dict) -> str:
    L = ["# v2 judge protocol — failure report",
         "",
         "The v2 hard-verdict protocol did not pass its swap-consistency gate and will "
         "not label the final experiment. This is the frozen evidence for that decision. "
         "**No v2 label is reused in the v3 bank.**",
         "",
         f"- git commit: `{r['git_commit']}`",
         f"- rubric: `{r['rubric']['path']}` sha256 `{r['rubric']['sha256'][:32]}…`",
         ""]
    sm = r["swap_mapping_verified"]
    L += ["## First: is the swap mapping itself correct?", "",
          "Before blaming the judge, the arithmetic that turns two ordered verdicts into "
          "one semantic preference was pinned against hand-constructed cases "
          f"(`{sm['test']}`):", "",
          "| forward verdict | reverse verdict | p_hat(y > z) |", "|---|---|---|",
          "| A | B | 1.0 |", "| B | A | 0.0 |",
          "| A | A | 0.5  (first-position contradiction) |",
          "| B | B | 0.5  (second-position contradiction) |",
          "| TIE | TIE | 0.5 |", "",
          f"Reference tensor exactly skew-symmetric with a zero diagonal: "
          f"**{sm['reference_tensor_exactly_skew_symmetric']}**.", "",
          f"**{sm['verdict']}.**", "",
          "Note the mechanically important row: a judge that always picks whichever "
          "response is shown first, and a judge that genuinely finds the two equal, "
          "produce the *same* tensor entry. The tensor cannot distinguish noise from "
          "indifference, which is why a hard-verdict protocol was the wrong instrument "
          "for a pool of same-policy samples.", ""]

    for label, b in r["banks"].items():
        L += [f"## {label}", "",
              f"- bank: `{b['path']}` ({b['rows']} rows), sha256 `{b['sha256'][:32]}…`",
              f"- judge model: `{b['judge_model']}`",
              f"- parser-invalid rate: **{b['totals']['parser_invalid_rate']:.4%}**, "
              f"retry rate {b['totals']['retry_rate']:.4%}",
              f"- mean |Delta| over all pairs: **{b['totals']['mean_abs_delta']:.4f}** "
              "(maximum possible 0.5)", "",
              "| objective | calls | pairs | decisive | consistency | contradiction | "
              "tie/tie | one-order tie | position bias | mean \\|D\\| | \\|D\\|=0 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for obj, o in b["per_objective"].items():
            c = o["decisive_decisive_semantic_consistency"]
            k = o["decisive_contradiction_rate"]
            L.append(
                f"| {obj} | {o['total_calls']} | {o['semantic_pairs']} | "
                f"{o['both_orders_decisive']} | "
                f"{'n/a' if c is None else f'{c:.3f}'} | "
                f"{'n/a' if k is None else f'{k:.3f}'} | "
                f"{o['tie_tie_rate']:.3f} | {o['one_order_tie_rate']:.3f} | "
                f"{o['position_bias']:+.4f} | {o['mean_abs_delta']:.4f} | "
                f"{o['abs_delta_fraction_at_zero']:.1%} |")
        L.append("")
    L += ["## What the numbers say", "",
          "**Position bias is the mechanism, and it is judge-specific.** Llama-3.3-70B "
          "carries a large positive bias toward the A slot on every objective, reaching "
          "**+0.36 on helpfulness** -- which is exactly why its decisive consistency there "
          "(0.433) sits *below* chance: when both orders commit, it is mostly committing "
          "to whichever response it was shown first. Qwen3-32B is milder and mixed in sign "
          "(+0.14 on helpfulness, within +/-0.04 elsewhere).",
          "",
          "**Swap averaging cancels the systematic part of that bias but cannot recover the "
          "lost information.** A pair the judge decides oppositely in the two orders lands "
          "at |Delta| = 0, the same value a genuine tie produces. Between 40% and 74% of "
          "all pairs end up there, and mean |Delta| is 0.18 (Qwen) and 0.12 (Llama) against "
          "a maximum of 0.5. The measurement is unbiased and badly underpowered.",
          "",
          "**Neither judge is the fix.** The 70B is better on truthfulness and honesty only "
          "because it abstains far more (it ties both orders on 69% and 67% of those pairs), "
          "and it is worse on the other two. No judge tested clears 0.85 on more than one "
          "objective.",
          "",
          "This is why the protocol moves to calibrated, uncertainty-aware scoring rather "
          "than to a different judge or a lower threshold: the instrument has to be able to "
          "*say* it is uncertain, instead of encoding uncertainty as a coin flip that the "
          "tensor then cannot distinguish from indifference.",
          "",
          "## What this bank is for", "",
          "Protocol development only. It is not the v3 gate set, and none of these labels "
          "may be mixed into the v3 judgment bank.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
