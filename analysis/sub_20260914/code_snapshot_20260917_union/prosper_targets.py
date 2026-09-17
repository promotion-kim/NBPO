"""PROSPER targets: Algorithm 1 step 7, with the appendix's leave-two-out banks.

From the paper's Algorithm 1 the policy step is a squared loss whose target for
the log-ratio difference is eta * (g_hat(z) - g_hat(z')), with

    k_hat(x) = argmin_k -beta log Z_hat^k(x)
    Z_hat^k  = (1/M) sum_j exp( -(1/(M beta)) sum_i P^k(y_i > y'_j | x) )
    g_hat(z) = sum_j P^{k_hat}(z > y'_j) w_j / sum_j w_j,
               w_j = exp( -(1/(M beta)) sum_i P^{k_hat}(y_i > y'_j) )

The appendix fixes the estimator: for each of the 64 learner/reference fitting
pairs (i, j), the criterion and the gradient are estimated from the OTHER seven
learner and seven reference occurrences, so M = 7 and both banks exclude the two
responses being fitted. g_hat at the reference response needs
reference-reference comparisons, which is why the judging graph carried the
reference triangle; it is read back here rather than approximated.

Everything is checked before a single GPU is touched: that the banks really
exclude i and j, that the softmax weights sum to one, that every target is
finite, and that the row count is exactly 64 per surviving prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SUB = Path("/work/sub_20260914")
UF = Path("/work/uf4_20260910")
POOL = 8
K = 4
ITEMS = ["item%d" % k for k in range(K)]


def file_hash(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def read_blocks(tag, shards):
    """prompt -> (cross[K,8,8], reftri[K,8,8]) of order-balanced probabilities."""
    obs = defaultdict(dict)
    for shard in range(shards):
        d = SUB / "pool_judgments" / tag / ("shard%d" % shard)
        for path in sorted(d.glob("chunk*.jsonl")):
            with path.open() as stream:
                for line in stream:
                    r = json.loads(line)
                    if r["status"] != "ok":
                        continue
                    key = (r["prompt_id"], r["rubric"], r["role_i"], r["i"],
                           r["role_j"], r["j"])
                    obs[key][r["order"]] = float(r["value_for_i"])

    def resolve(pid, rubric, ri, i, rj, j):
        got = obs.get((pid, rubric, ri, i, rj, j))
        if got is None or 0 not in got or 1 not in got:
            return None
        return 0.5 * (got[0] + got[1])

    out, dropped = {}, defaultdict(int)
    for pid in sorted({k[0] for k in obs}):
        cross = np.zeros((K, POOL, POOL))
        reftri = np.zeros((K, POOL, POOL))
        ok = True
        for k, rubric in enumerate(ITEMS):
            for i in range(POOL):
                for j in range(POOL):
                    p = resolve(pid, rubric, "learner", i, "comparator", j)
                    if p is None:
                        ok = False; dropped["cross"] += 1; break
                    cross[k, i, j] = p
                if not ok:
                    break
            if not ok:
                break
            for a, b in itertools.combinations(range(POOL), 2):
                p = resolve(pid, rubric, "comparator", a, "comparator", b)
                if p is None:
                    ok = False; dropped["reference_triangle"] += 1; break
                reftri[k, a, b] = p
                reftri[k, b, a] = 1.0 - p
            if not ok:
                break
        if ok:
            out[pid] = (cross, reftri)
    return out, dict(dropped)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="uw1_psc")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--splits", default="uw_v1p")
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--out", default="uw1_prosper")
    args = ap.parse_args()

    blocks, dropped = read_blocks(args.tag, args.shards)
    keep = {}
    for split in ("policy_train", "policy_dev"):
        ids = [json.loads(l)["prompt_id"] for l in
               (UF / "splits" / args.splits / ("%s.jsonl" % split)).open() if l.strip()]
        keep[split] = [p for p in ids if p in blocks]

    M = POOL - 1
    checks = {"bank_excludes_fitting_pair": True, "weights_sum_to_one": True,
              "targets_finite": True}
    rows = {split: [] for split in keep}
    chosen_items = defaultdict(list)
    for split, ids in keep.items():
        for pid in ids:
            cross, reftri = blocks[pid]
            for i in range(POOL):
                for j in range(POOL):
                    learners = [a for a in range(POOL) if a != i]
                    refs = [b for b in range(POOL) if b != j]
                    if i in learners or j in refs:
                        checks["bank_excludes_fitting_pair"] = False
                    # Eq. 11 over the leave-two-out banks
                    scores = []
                    for k in range(K):
                        inner = cross[k][np.ix_(learners, refs)].sum(axis=0)  # sum_i
                        w = np.exp(-inner / (M * args.beta))
                        z = w.mean()
                        scores.append(-args.beta * np.log(max(z, 1e-300)))
                    khat = int(np.argmin(scores))
                    inner = cross[khat][np.ix_(learners, refs)].sum(axis=0)
                    w = np.exp(-inner / (M * args.beta))
                    total = w.sum()
                    if not np.isfinite(total) or total <= 0:
                        checks["weights_sum_to_one"] = False
                        continue
                    w = w / total
                    if abs(float(w.sum()) - 1.0) > 1e-9:
                        checks["weights_sum_to_one"] = False
                    g_learner = float(np.dot(cross[khat, i, refs], w))
                    g_reference = float(np.dot(reftri[khat, j, refs], w))
                    target = args.eta * (g_learner - g_reference)
                    if not np.isfinite(target):
                        checks["targets_finite"] = False
                        continue
                    rows[split].append({"prompt_id": pid, "learner_index": i,
                                        "reference_index": j, "objective": khat,
                                        "g_learner": round(g_learner, 6),
                                        "g_reference": round(g_reference, 6),
                                        "ronpo_target": target})
                    chosen_items[pid].append(khat)

    out = UF / "targets" / args.out
    out.mkdir(parents=True, exist_ok=True)
    report = {"algorithm": "PROSPER Algorithm 1 step 7 with Eq. 11 criterion",
              "estimator": "M = %d leave-two-out banks per fitting pair" % M,
              "beta": args.beta, "eta": args.eta,
              "judgment_tag": args.tag, "splits": args.splits,
              "prompts_with_complete_blocks": len(blocks),
              "dropped_blocks": dropped,
              "checks": checks}
    for split, data in rows.items():
        path = out / ("%s.jsonl" % split)
        with path.open("w") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")
        per_prompt = defaultdict(int)
        for r in data:
            per_prompt[r["prompt_id"]] += 1
        t = np.array([r["ronpo_target"] for r in data]) if data else np.zeros(0)
        report[split] = {
            "prompts": len(per_prompt), "rows": len(data),
            "rows_per_prompt": sorted(set(per_prompt.values())),
            "target": {"mean": round(float(t.mean()), 6) if len(t) else None,
                       "sd": round(float(t.std()), 6) if len(t) else None,
                       "min": round(float(t.min()), 6) if len(t) else None,
                       "max": round(float(t.max()), 6) if len(t) else None},
            "objective_counts": {str(k): int(sum(1 for r in data if r["objective"] == k))
                                 for k in range(K)},
            "sha256": file_hash(path)}
    (out / "targets_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    for name, ok in checks.items():
        if not ok:
            raise SystemExit("check failed: %s" % name)
    for split in rows:
        if report[split]["rows_per_prompt"] != [POOL * POOL]:
            raise SystemExit("%s has %s rows per prompt, expected %d"
                             % (split, report[split]["rows_per_prompt"], POOL * POOL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
