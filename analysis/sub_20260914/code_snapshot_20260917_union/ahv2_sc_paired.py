"""Paired whole-prompt bootstrap of the style-controlled win-rate DIFFERENCE.

Two arms' style-controlled intervals can overlap heavily and still differ
reliably, because they are read on the SAME 750 prompts and the prompt effect is
common to both. The unpaired intervals are therefore the wrong test for "does
this arm beat that one", and this file computes the right one: resample prompts
once per replicate, refit BOTH arms' style regressions on that resample, and
take the difference of the two style-free intercepts.

A replicate in which either fit fails to solve is counted and excluded from
both sides, never silently dropped from one. The interval is the 2.5/97.5
quantiles of the 2,000 differences, and the reported p-value is the two-sided
bootstrap proportion crossing zero -- an interval containing zero is reported as
containing zero, never as equivalence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, "/work/sub_20260914/code")
from ahv2_style_control import ORDERS, fit, make_length, normalized_difference, style


def load(judged: Path, baseline: Path, panel: Path, length=None, dropped=None):
    """Both answers counted by the SAME `length`, and both orders or neither.

    The two rules are the point of this loader. Measuring the arm in tokenizer
    tokens and the baseline in whitespace words made identical answers differ by
    .44 on the length feature, which biases the intercept the control exists to
    isolate; and a prompt present in one presentation order only would enter with
    a first-position bias the raw figure does not carry.
    """
    if length is None:
        # a caller that does not care about the unit still gets ONE unit on both
        # sides; the defect this guards against was two different units, not the
        # choice between them
        length, _ = make_length("words", None)
    complete = json.loads((judged / "complete.json").read_text())
    arm = {}
    for line in Path(complete["arm_responses"]).open():
        r = json.loads(line)
        arm[r["prompt_id"]] = style(r["response"], length)
    uid_of = {}
    for line in panel.open():
        r = json.loads(line)
        uid_of[r["prompt_id"]] = r["uid"]
    by_uid = {}
    for line in baseline.open():
        r = json.loads(line)
        text = r["messages"][-1]["content"] if "messages" in r else r.get("response", "")
        if isinstance(text, dict):
            text = text.get("answer", "")
        by_uid[r["uid"]] = style(text, length)
    base = {pid: by_uid[uid] for pid, uid in uid_of.items() if uid in by_uid}
    if dropped is None:
        dropped = {}
    seen, single_order = {}, 0
    for line in (judged / "verdicts.jsonl").open():
        v = json.loads(line)
        if v.get("status") != "ok":
            continue
        pid = v["prompt_id"]
        if pid in arm and pid in base:
            seen.setdefault(pid, {})[int(v["order"])] = float(v["value_for_arm"])
    rows = []
    for pid, byorder in seen.items():
        if len(byorder) != ORDERS:
            single_order += 1
            continue
        d = normalized_difference(arm[pid], base[pid])
        for order in sorted(byorder):
            rows.append((pid, byorder[order], d))
    # reported through the caller's dict rather than a third return value, so an
    # existing two-value caller keeps working and still gets the both-order rule
    dropped["single_order_prompts"] = single_order
    return complete["arm"], rows


def assemble(rows, pids):
    index = {p: i for i, p in enumerate(pids)}
    buckets = [[] for _ in pids]
    y = np.empty(len(rows))
    D = np.empty((len(rows), 4))
    for i, (pid, val, feat) in enumerate(rows):
        y[i] = val
        D[i] = feat
        buckets[index[pid]].append(i)
    X = np.column_stack([np.ones(len(rows)), D])
    return y, X, buckets


def sc(y, X, idx):
    """The style-free win rate, or None if the fit did not converge."""
    got = fit(y[idx], X[idx])
    if got is None or not got[1]["converged"]:
        return None
    return 1.0 / (1.0 + math.exp(-got[0][0]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="judged directory of the arm under test")
    ap.add_argument("--b", required=True, help="judged directory of the comparator")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--length-unit", default="tokens", choices=("tokens", "words"))
    ap.add_argument("--tokenizer", default="/work/models/bases/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    length, length_meta = make_length(args.length_unit, args.tokenizer)
    probe = style("one two three four five six seven", length)
    if float(normalized_difference(probe, probe)[0]) != 0.0:
        raise SystemExit("length counter is not self-consistent")

    baseline, panel = Path(args.baseline), Path(args.panel)
    drop_a, drop_b = {}, {}
    name_a, rows_a = load(Path(args.a), baseline, panel, length, drop_a)
    name_b, rows_b = load(Path(args.b), baseline, panel, length, drop_b)
    # the paired unit is a prompt scored in BOTH runs
    pids = sorted({r[0] for r in rows_a} & {r[0] for r in rows_b})
    rows_a = [r for r in rows_a if r[0] in set(pids)]
    rows_b = [r for r in rows_b if r[0] in set(pids)]
    ya, Xa, ba = assemble(rows_a, pids)
    yb, Xb, bb = assemble(rows_b, pids)

    pa, pb = sc(ya, Xa, np.arange(len(ya))), sc(yb, Xb, np.arange(len(yb)))
    if pa is None or pb is None:
        raise SystemExit("a point fit did not solve")

    rng = np.random.default_rng(args.seed)
    n = len(pids)
    diffs, failed = [], 0
    for _ in range(args.replicates):
        pick = rng.integers(0, n, n)
        ia = np.concatenate([ba[k] for k in pick])
        ib = np.concatenate([bb[k] for k in pick])
        sa, sb = sc(ya, Xa, ia), sc(yb, Xb, ib)
        if sa is None or sb is None:
            failed += 1
            continue
        diffs.append(sa - sb)
    d = np.asarray(diffs)
    lo, hi = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
    # two-sided bootstrap p: how often the sign disagrees with the point estimate
    point = pa - pb
    p_two = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    out = {
        "arm": name_a, "comparator": name_b,
        "metric": "style_controlled_win_rate_local_implementation",
        "style_controlled": {name_a: pa, name_b: pb},
        "difference": point,
        "ci95": [lo, hi],
        "p_two_sided_bootstrap": min(p_two, 1.0),
        "contains_zero": bool(lo <= 0.0 <= hi),
        "reading": ("an interval containing zero means this comparison does not separate the "
                    "two arms; it is not evidence that they are equal"),
        "paired_prompts": n,
        "length_counter": length_meta,
        "single_order_prompts_dropped": {
            name_a: drop_a.get("single_order_prompts"),
            name_b: drop_b.get("single_order_prompts")},
        "bootstrap": {"replicates": args.replicates, "unit": "prompt",
                      "refit_both_arms_per_replicate": True,
                      "failed_replicates": failed, "seed": args.seed},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("arm", "comparator", "difference", "ci95",
                                          "p_two_sided_bootstrap", "contains_zero")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
