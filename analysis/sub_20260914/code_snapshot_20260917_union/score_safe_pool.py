"""Direct judge verdicts -> the solver's score tensors, in the UF-4 npz format.

The solver's loader expects, per prompt:

    A_policy[K, 8, 8]   learner i against comparator j
    A_ref[K, 8, 8]      comparator i against comparator j
    r_bt[K, 2, 8]       a scalar score per occurrence, learners then comparators

so the 92 judged pairs map onto it exactly: 64 cross pairs fill A_policy and 28
reference pairs fill A_ref. A_policy is NOT antisymmetrised -- learners and
comparators are different sets, and forcing A + A^T = 0 across them would
invent comparisons that were never made. A_ref is antisymmetrised with an
exactly zero diagonal, which is the reference bank's own matrix.

    p_hat_k(u, v) = 1/2 [ mean over u-first draws + mean over v-first draws ]
    A[k, u, v]    = p_hat_k(u, v) - 1/2

r_bt is a per-prompt Bradley-Terry projection onto the SAME observed
comparisons, with a sum-to-zero gauge and a fixed ridge. It is a finite
projection used by the BT-representation arms and by scalarized DPO; it is not
a globally trained reward model and is not described as one.

Missingness is handled at the prompt level: a pair needs at least one parsed
draw in each presentation order, and a prompt that leaves any scheduled pair
unresolved is excluded from the tensor set and counted. No invalid verdict
becomes a tie or a loss, and no candidate is deleted to renormalise anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

UF = Path("/work/uf4_20260910")
SUB = Path("/work/sub_20260914")


def digest_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def file_hash(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def bt_fit(pairs, n, ridge=1e-3, iters=5000):
    """Sum-to-zero BT scores from observed pairwise win probabilities.

    pairs: list of (u, v, p_uv) with p_uv in [0, 1]. Minimises the weighted
    logistic NLL plus ridge*||r||^2 by gradient descent with backtracking, which
    is enough for a 16-node graph and keeps the gauge explicit.

    The iteration cap is 5,000 because 200 left 1,436 of 4,000 fits short of the
    1e-6 gradient tolerance. Probing 24 of them, 5,000 iterations converged all
    24 and moved the scores by a median of 0.0 and at most 0.424 on a score
    range of about 6.15, i.e. 2.9% of the range in the worst case. Small, but
    the fit feeds the BT-projected Nash targets and the scalarized-DPO labels,
    and removing it costs a few CPU minutes.
    """
    r = np.zeros(n)
    if not pairs:
        return r, {"converged": True, "iterations": 0, "final_grad_inf": 0.0}
    u = np.array([p[0] for p in pairs])
    v = np.array([p[1] for p in pairs])
    y = np.clip(np.array([p[2] for p in pairs]), 1e-6, 1 - 1e-6)

    def nll(r):
        z = r[u] - r[v]
        # -[y log sigma(z) + (1-y) log sigma(-z)] summed, plus ridge
        return float(np.sum(np.logaddexp(0.0, -z) * y + np.logaddexp(0.0, z) * (1 - y))
                     + ridge * float(r @ r))

    def grad(r):
        z = r[u] - r[v]
        s = 1.0 / (1.0 + np.exp(-z))
        g = np.zeros_like(r)
        np.add.at(g, u, s - y)
        np.add.at(g, v, y - s)
        return g + 2.0 * ridge * r

    step, value = 1.0, nll(r)
    for it in range(iters):
        g = grad(r)
        g = g - g.mean()                       # stay in the sum-to-zero gauge
        if np.max(np.abs(g)) < 1e-9:
            return r - r.mean(), {"converged": True, "iterations": it,
                                  "final_grad_inf": float(np.max(np.abs(g)))}
        while step > 1e-12:
            trial = r - step * g
            trial = trial - trial.mean()
            new = nll(trial)
            if new <= value:
                r, value = trial, new
                step *= 1.5
                break
            step *= 0.5
        else:
            break
    g = grad(r)
    g = g - g.mean()
    return r - r.mean(), {"converged": bool(np.max(np.abs(g)) < 1e-6),
                          "iterations": iters, "final_grad_inf": float(np.max(np.abs(g)))}


def load_verdicts(tag, shards):
    """(prompt, rubric, role_i, i, role_j, j) -> {order: [values]}"""
    table = defaultdict(lambda: defaultdict(list))
    counts = Counter()
    sources = {}
    for shard in range(shards):
        directory = SUB / "pool_judgments" / tag / ("shard%d" % shard)
        if not directory.exists():
            continue
        for path in sorted(directory.glob("chunk*.jsonl")):
            manifest_path = directory / (path.stem + ".manifest.json")
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if file_hash(path) != manifest["sha256"]:
                    raise ValueError("verdict chunk hash mismatch: %s" % path)
            sources[str(path)] = file_hash(path)
            with path.open() as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    counts["read"] += 1
                    if r["status"] != "ok":
                        counts["unparsed"] += 1
                        continue
                    key = (r["prompt_id"], r["rubric"], r["role_i"], r["i"],
                           r["role_j"], r["j"])
                    table[key][r["order"]].append(float(r["value_for_i"]))
    return table, counts, sources


def p_hat(entry):
    if entry is None or 0 not in entry or 1 not in entry:
        return None
    return 0.5 * (float(np.mean(entry[0])) + float(np.mean(entry[1])))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="pool_judgments tag")
    ap.add_argument("--judge-shards", type=int, default=4)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--pool-shards", type=int, default=4)
    ap.add_argument("--out-name", required=True, help="scores/<out-name>")
    ap.add_argument("--out-shards", type=int, default=4)
    ap.add_argument("--objectives", nargs="+",
                    default=["helpfulness", "harmlessness"])
    ap.add_argument("--pool-per-role", type=int, default=8)
    ap.add_argument("--prompts-per-chunk", type=int, default=250)
    args = ap.parse_args()

    table, counts, sources = load_verdicts(args.tag, args.judge_shards)
    prompts = sorted({k[0] for k in table})
    I = args.pool_per_role
    K = len(args.objectives)

    complete, excluded = [], []
    tensors = {}
    bt_reports = Counter()
    for pid in prompts:
        A_pol = np.zeros((K, I, I))
        A_ref = np.zeros((K, I, I))
        r_bt = np.zeros((K, 2, I))
        ok = True
        missing = []
        for k, rub in enumerate(args.objectives):
            bt_pairs = []
            for i in range(I):
                for j in range(I):
                    p = p_hat(table.get((pid, rub, "learner", i, "comparator", j)))
                    if p is None:
                        ok = False
                        missing.append(["learner", i, "comparator", j, rub])
                        continue
                    A_pol[k, i, j] = p - 0.5
                    bt_pairs.append((i, I + j, p))
            for i in range(I):
                for j in range(i + 1, I):
                    p = p_hat(table.get((pid, rub, "comparator", i, "comparator", j)))
                    if p is None:
                        ok = False
                        missing.append(["comparator", i, "comparator", j, rub])
                        continue
                    A_ref[k, i, j] = p - 0.5
                    A_ref[k, j, i] = 0.5 - p
                    bt_pairs.append((I + i, I + j, p))
            scores, report = bt_fit(bt_pairs, 2 * I)
            r_bt[k, 0] = scores[:I]
            r_bt[k, 1] = scores[I:]
            bt_reports["converged" if report["converged"] else "not_converged"] += 1
        if not ok:
            excluded.append({"prompt_id": pid, "unresolved_pairs": missing[:6],
                             "unresolved_count": len(missing)})
            continue
        idx = np.arange(I)
        A_ref[:, idx, idx] = 0.0
        if np.abs(A_ref + np.swapaxes(A_ref, -1, -2)).max() > 1e-12:
            raise AssertionError("reference matrix is not antisymmetric for %s" % pid)
        if np.abs(A_pol).max() > 0.5 + 1e-12 or np.abs(A_ref).max() > 0.5 + 1e-12:
            raise AssertionError("payoff outside [-1/2, 1/2] for %s" % pid)
        tensors[pid] = (A_pol, A_ref, r_bt)
        complete.append(pid)

    teacher = {
        "kind": "direct pairwise LLM judgment",
        "judge": str(UF / "assets/Qwen3-14B"),
        "judge_revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
        "revision_is_a_declaration": True,
        "objectives": list(args.objectives),
        "orders": 2, "draws_per_order": 2,
        "not_a_score_model": ("this key exists because the UF-4 loader requires it; "
                              "the teacher here is a judge on raw text, not the "
                              "ModernBERT score-induced teacher of the UF-4 campaign"),
        "verdict_aggregation": "p_hat = mean over both presentation orders, ties 0.5",
        "judgment_sources_sha256": sources}
    bt_teacher = {
        "kind": "per-prompt finite Bradley-Terry projection on the observed pairs",
        "gauge": "scores sum to zero within a prompt",
        "ridge": 1e-3,
        "observed_pairs": "64 learner-comparator and 28 comparator-comparator",
        "not_a_trained_reward_model": True,
        "fit_status": dict(bt_reports)}

    out_root = UF / "scores" / args.out_name
    shard_of = {pid: index % args.out_shards for index, pid in enumerate(complete)}
    written = {}
    for shard in range(args.out_shards):
        pids = [p for p in complete if shard_of[p] == shard]
        directory = out_root / ("shard%d" % shard)
        directory.mkdir(parents=True, exist_ok=True)
        chunks = [pids[i:i + args.prompts_per_chunk]
                  for i in range(0, len(pids), args.prompts_per_chunk)] or [[]]
        for chunk_index, chunk in enumerate(chunks):
            if not chunk:
                continue
            path = directory / ("chunk%04d.npz" % chunk_index)
            A = np.stack([tensors[p][0] for p in chunk], axis=1)
            Aref = np.stack([tensors[p][1] for p in chunk], axis=1)
            Rbt = np.stack([tensors[p][2] for p in chunk], axis=1)
            np.savez(path, prompt_ids=np.array(chunk), A_policy=A, A_ref=Aref, r_bt=Rbt)
            (directory / (path.stem + ".manifest.json")).write_text(json.dumps(
                {"prompts": len(chunk), "sha256": file_hash(path),
                 "shapes": {"A_policy": list(A.shape), "A_ref": list(Aref.shape),
                            "r_bt": list(Rbt.shape)}}, indent=2) + "\n")
        (directory / ("complete_shard%d.json" % shard)).write_text(json.dumps(
            {"shard": shard, "prompts": len(pids),
             "gpm_teacher": teacher, "bt_teacher": bt_teacher,
             "reference_construction": "comparator_self_bank",
             "objectives": list(args.objectives)}, indent=2) + "\n")
        written[shard] = len(pids)

    report = {"tag": args.tag, "pool": args.pool, "out_name": args.out_name,
              "verdicts_read": counts["read"], "unparsed": counts["unparsed"],
              "parse_rate": (counts["read"] - counts["unparsed"]) / max(counts["read"], 1),
              "prompts_seen": len(prompts), "prompts_complete": len(complete),
              "prompts_excluded": len(excluded),
              "exclusion_rate": len(excluded) / max(len(prompts), 1),
              "excluded_examples": excluded[:5],
              "shards": written, "bt_fit": dict(bt_reports),
              "teacher_hash": digest_obj({k: v for k, v in teacher.items()
                                          if k != "judgment_sources_sha256"}),
              "source_sha256": file_hash(__file__),
              "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out_root / "score_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("verdicts_read", "parse_rate", "prompts_seen", "prompts_complete",
                       "prompts_excluded", "shards", "bt_fit")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
