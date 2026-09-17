"""Build UW game tensors from the union judging graph.

The recorded PROSPER-setting scorer is a SELF-PLAY construction, and says so in
its own docstring: one bank of eight responses, A_policy antisymmetric among
them, and A_ref a copy of A_policy. The union contract asks for something
different -- a learner bank and a separate reference bank -- so this builds:

  A_policy[k,i,j] = P_k(learner_i > learner_j) - 1/2      from the learner triangle
  A_ref[k,i,j]    = P_k(learner_i > reference_j) - 1/2    from the 8x8 cross block

A_policy is antisymmetric by construction and is checked. A_ref is NOT
antisymmetric: its two indices name different banks, and its diagonal need not
be zero, which the contract states explicitly. The reference triangle is not
part of either matrix; it is kept as the reference-disagreement diagnostic the
contract asks for.

Both orders must be valid for a pair to count, exactly as declared: with
value_for_i already reversed for the swapped order, the order-balanced estimate
is their mean, which equals 1/2[v_ij/4 + 1 - v_ji/4]. A pair of identical text
is 0.5 by convention and logged as such. A prompt enters only when every
learner pair and every cross pair is resolved for all four items; partial
prompts are dropped with their reason, never filled in.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

UF = Path("/work/uf4_20260910")
SUB = Path("/work/sub_20260914")
POOL = 8
K = 4
ITEMS = ["item%d" % k for k in range(K)]


def file_hash(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_pool(pool_name, shards):
    """prompt_id -> role -> index -> response text sha, for the identity check."""
    out = defaultdict(lambda: defaultdict(dict))
    settings = []
    for shard in range(shards):
        d = UF / "pools" / pool_name / ("shard%d" % shard)
        settings.append(json.loads((d / "settings.json").read_text()))
        for path in sorted(d.glob("chunk*.jsonl")):
            manifest = json.loads((d / (path.stem + ".manifest.json")).read_text())
            if file_hash(path) != manifest["sha256"]:
                raise SystemExit("pool chunk hash mismatch: %s" % path)
            with path.open() as stream:
                for line in stream:
                    e = json.loads(line)
                    out[e["prompt_id"]][e["role"]][e["sample_index"]] = e["response_sha256"]
    return out, settings


SUPERSEDED = (
    "union_score_uw.py is superseded by union_score_panel.py and refuses to run.\n"
    "\n"
    "It wires A_policy to the learner triangle and A_ref to the learner-by-reference\n"
    "cross block, keeping the reference triangle only as a scalar diagnostic. That is\n"
    "not the finite game Section 5.2 defines, and it is not what\n"
    "compute_disagreement_point documents its argument to be: with every learner tied\n"
    "to every other learner, every reference tied to every other reference, and every\n"
    "learner beating every reference at .75, this wiring turns a surplus of +.25 into\n"
    "-.25. The surplus is what finite-pool feasibility tests, so a score directory\n"
    "written here cannot be solved into the declared problem.\n"
    "\n"
    "Use:\n"
    "  python3 union_score_panel.py --objectives 4 --tag <tag> --pool <pool> --out <dir>\n"
    "\n"
    "It reproduces this file's own UW tensors exactly where they agree, writes A_LL,\n"
    "A_LR and A_RR each under its own name, and stamps tensor_role_schema so the\n"
    "solver can tell a corrected shard from a pre-fix one. This file is kept only as\n"
    "the provenance of scores written before 2026-09-18; run it with\n"
    "--i-know-this-is-the-superseded-scorer if you are deliberately reproducing those.\n"
)


def refuse_unless_acknowledged():
    import sys
    flag = "--i-know-this-is-the-superseded-scorer"
    if flag in sys.argv:
        sys.argv.remove(flag)
        sys.stderr.write("WARNING: running the superseded UW scorer on purpose.\n")
        return
    raise SystemExit(SUPERSEDED)


def main():
    refuse_unless_acknowledged()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="uw1_psc")
    ap.add_argument("--judge-shards", type=int, default=4)
    ap.add_argument("--pool", default="uw1")
    ap.add_argument("--pool-shards", type=int, default=4)
    ap.add_argument("--out", default="uw1")
    ap.add_argument("--shards", type=int, default=4)
    args = ap.parse_args()

    pool, pool_settings = load_pool(args.pool, args.pool_shards)
    status = Counter()
    # (pid, rubric, role_i, i, role_j, j) -> {order: value_for_i}
    obs = defaultdict(dict)
    sources = []
    for shard in range(args.judge_shards):
        d = SUB / "pool_judgments" / args.tag / ("shard%d" % shard)
        if not (d / "complete.json").exists():
            raise SystemExit("judging shard %d is not complete" % shard)
        for path in sorted(d.glob("chunk*.jsonl")):
            sources.append({"path": str(path), "sha256": file_hash(path)})
            with path.open() as stream:
                for line in stream:
                    r = json.loads(line)
                    status[r["status"]] += 1
                    if r["status"] != "ok":
                        continue
                    key = (r["prompt_id"], r["rubric"], r["role_i"], r["i"],
                           r["role_j"], r["j"])
                    obs[key][r["order"]] = float(r["value_for_i"])

    learner_pairs = list(itertools.combinations(range(POOL), 2))
    cross_pairs = [(i, j) for i in range(POOL) for j in range(POOL)]

    def resolve(pid, rubric, role_i, i, role_j, j):
        """Order-balanced probability, or None when an order is missing."""
        got = obs.get((pid, rubric, role_i, i, role_j, j))
        if got is None or 0 not in got or 1 not in got:
            return None
        return 0.5 * (got[0] + got[1])

    tensors, identity_ties, refdis = {}, Counter(), {}
    dropped = Counter()
    prompts = sorted({k[0] for k in obs})
    for pid in prompts:
        if pid not in pool:
            dropped["prompt_absent_from_pool"] += 1
            continue
        A = np.zeros((K, POOL, POOL))
        R = np.zeros((K, POOL, POOL))
        ok = True
        for k, rubric in enumerate(ITEMS):
            for i, j in learner_pairs:
                p = resolve(pid, rubric, "learner", i, "learner", j)
                if p is None:
                    ok = False; dropped["learner_pair_unresolved"] += 1; break
                A[k, i, j] = p - 0.5
                A[k, j, i] = 0.5 - p
            if not ok:
                break
            for i, j in cross_pairs:
                p = resolve(pid, rubric, "learner", i, "comparator", j)
                if p is None:
                    ok = False; dropped["cross_pair_unresolved"] += 1; break
                if (pool[pid]["learner"].get(i) ==
                        pool[pid]["comparator"].get(j) is not None):
                    identity_ties[pid] += 1
                R[k, i, j] = p - 0.5
            if not ok:
                break
        if not ok:
            continue
        idx = np.arange(POOL)
        A[:, idx, idx] = 0.0
        if np.abs(A + np.swapaxes(A, -1, -2)).max() > 1e-12:
            raise SystemExit("learner payoff not antisymmetric for %s" % pid)
        if max(np.abs(A).max(), np.abs(R).max()) > 0.5 + 1e-12:
            raise SystemExit("payoff outside [-0.5, 0.5] for %s" % pid)
        # reference disagreement: the reference triangle, kept as a diagnostic
        vals = []
        for k, rubric in enumerate(ITEMS):
            for i, j in learner_pairs:
                p = resolve(pid, rubric, "comparator", i, "comparator", j)
                if p is not None:
                    vals.append(abs(p - 0.5))
        refdis[pid] = float(np.mean(vals)) if vals else None
        tensors[pid] = (A, R)

    pids = sorted(tensors)
    if not pids:
        raise SystemExit("no prompt survived scoring")
    out = UF / "scores" / args.out
    out.mkdir(parents=True, exist_ok=True)
    teacher = {
        "judge": "/work/uf4_20260910/assets/Qwen3-14B",
        "judge_revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
        "judgment_tag": args.tag,
        "template": "PROSPER Figure 4 PSC five-point single-check",
        "scale": "verdict in {0..4} for the FIRST response; p = verdict/4",
        "draws_per_order": 1,
        "orders": "both, swapped verdict reversed before averaging",
        "graph": "cross 8x8 + learner C(8,2) + reference C(8,2) per item",
        "items_per_prompt": K,
    }
    per = (len(pids) + args.shards - 1) // args.shards
    written = []
    for s in range(args.shards):
        chunk = pids[s * per:(s + 1) * per]
        if not chunk:
            continue
        A = np.stack([tensors[p][0] for p in chunk], axis=1)
        R = np.stack([tensors[p][1] for p in chunk], axis=1)
        d = out / ("shard%d" % s)
        d.mkdir(parents=True, exist_ok=True)
        path = d / "chunk0000.npz"
        np.savez(path, prompt_ids=np.array(chunk), A_policy=A, A_ref=R)
        (d / "chunk0000.manifest.json").write_text(json.dumps(
            {"prompts": len(chunk), "sha256": file_hash(path),
             "shapes": {"A_policy": list(A.shape), "A_ref": list(R.shape)}},
            indent=1) + "\n")
        (d / ("complete_shard%d.json" % s)).write_text(json.dumps(
            {"shard": s, "prompts": len(chunk), "gpm_teacher": teacher,
             "bt_teacher": ("absent: the union contract uses direct order-balanced PSC "
                            "probabilities for these rows and forbids a scalar BT "
                            "projection, so none was fitted and none is written"),
             "reference_construction": ("independent reference bank: eight reference "
                                        "occurrences per prompt, so A_ref is a cross "
                                        "block and not a copy of A_policy")},
            indent=1) + "\n")
        written.append({"shard": s, "prompts": len(chunk),
                        "sha256": file_hash(path),
                        "shapes": {"A_policy": list(A.shape), "A_ref": list(R.shape)}})

    verdicts = sum(status.values())
    report = {
        "out": args.out, "judgment_tag": args.tag, "pool": args.pool,
        "construction": {
            "A_policy": "learner triangle, antisymmetric, diagonal zero",
            "A_ref": ("8x8 cross block learner-by-reference; NOT antisymmetric and its "
                      "diagonal need not be zero, per the contract"),
            "reference_triangle": "kept as the reference-disagreement diagnostic only",
            "not_self_play": ("the recorded PROSPER-setting scorer copies A_policy into "
                              "A_ref because that panel has one bank; this panel has two"),
        },
        "prompts_seen": len(prompts), "prompts_complete": len(pids),
        "retained_fraction": round(len(pids) / max(len(prompts), 1), 4),
        "dropped": dict(dropped),
        "verdicts_read": verdicts,
        "status_counts": dict(status),
        "parse_rate": round(status["ok"] / max(verdicts, 1), 6),
        "identity_text_pairs": {"prompts_with_any": len(identity_ties),
                                "total_pairs": int(sum(identity_ties.values())),
                                "convention": "0.5 by convention, logged not silent"},
        "reference_disagreement": {
            "mean_over_prompts": (round(float(np.mean([v for v in refdis.values()
                                                       if v is not None])), 6)
                                  if any(v is not None for v in refdis.values()) else None),
            "prompts_without_any": sum(1 for v in refdis.values() if v is None)},
        "shards": written,
        "pool_settings_agree": len({json.dumps({k: v for k, v in s.items()
                                                if k not in ("shard", "split_files_sha256",
                                                             "splits_dir", "source_sha256")},
                                               sort_keys=True)
                                    for s in pool_settings}) == 1,
        "judgment_sources": len(sources),
        "source_sha256": file_hash(__file__),
    }
    (out / "score_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("prompts_seen", "prompts_complete", "retained_fraction",
                       "dropped", "parse_rate", "identity_text_pairs",
                       "reference_disagreement")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
