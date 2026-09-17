# Generated from union_score_uw.py by make_panel_scorer.py -- do not edit by hand.
# source sha256 aefb140d0af0c027bffe4cb4ce805b195af77cd72b90496a6b43453a1b8bac82
# change: the hardcoded objective count K = 4 becomes the required flag
#         --objectives, so one scorer serves UW (4 items) and US/UT (2).
#         Nothing else differs; the UW scores already written are not
#         recomputed by this file.
"""Build UW game tensors from the union judging graph.

The recorded PROSPER-setting scorer is a SELF-PLAY construction, and says so in
its own docstring: one bank of eight responses, A_policy antisymmetric among
them, and A_ref a copy of A_policy. The union contract asks for something
different -- a learner bank and a separate reference bank -- so this builds:

  A_LL[k,i,j] = P_k(learner_i   > learner_j)   - 1/2   learner triangle
  A_LR[k,i,j] = P_k(learner_i   > reference_j) - 1/2   8x8 cross block
  A_RR[k,i,j] = P_k(reference_i > reference_j) - 1/2   reference triangle

and hands the NBPO solver

  A_policy = A_LR      the learner-versus-reference game the policy plays
  A_ref    = A_RR      the reference-versus-reference game its fallback d uses

An earlier version of this file wired A_policy to A_LL and A_ref to A_LR, and
kept the reference triangle only as a scalar mean-absolute-margin diagnostic.
That is not the finite game Section 5.2 defines, and it is not what
compute_disagreement_point documents its argument to be: "centered payoffs of
reference responses (as learner, index i) against reference comparators". The
mis-wiring is not cosmetic. With every learner tied to every other learner,
every reference tied to every other reference, and every learner beating every
reference at .75, the paper's wiring gives policy value +.25, disagreement 0 and
surplus +.25, while the old wiring gives 0, +.25 and surplus -.25. It flips the
sign of the surplus, and the surplus is what finite-pool feasibility tests.

A_LL is also structurally the wrong object for the policy game: it is forced
antisymmetric with a zero diagonal, so a symmetric strategy scores identically
zero against itself and the surplus carries almost no signal about the learner
bank. That is consistent with the near-zero target correlations the UW and US
rounds measured.

A_LL is still written, because it is the right tensor for single-objective pair
labels and build_panel_softlabels consumes it as such. Each tensor is stored
under its own name so a consumer cannot silently take one for another.

A_LL and A_RR are antisymmetric by construction and both are checked. A_LR is
NOT antisymmetric: its two indices name different banks and its diagonal need
not be zero. A prompt now enters only when the learner triangle, the cross block
AND the reference triangle are resolved for every item; a prompt missing the
reference triangle used to be retained and is now dropped with its reason,
because d cannot be computed without it.

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
K = None                    # set from --objectives; no default objective count
ITEMS = None


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--objectives", type=int, required=True,
                    help="declared objectives per prompt: 4 for UW, 2 for US and UT")
    ap.add_argument("--tag", default="uw1_psc")
    ap.add_argument("--judge-shards", type=int, default=4)
    ap.add_argument("--pool", default="uw1")
    ap.add_argument("--pool-shards", type=int, default=4)
    ap.add_argument("--out", default="uw1")
    ap.add_argument("--shards", type=int, default=4)
    args = ap.parse_args()
    global K, ITEMS
    if args.objectives < 1:
        raise SystemExit("--objectives must be positive")
    K = args.objectives
    ITEMS = ["item%d" % k for k in range(K)]

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
        A_LL = np.zeros((K, POOL, POOL))
        A_LR = np.zeros((K, POOL, POOL))
        A_RR = np.zeros((K, POOL, POOL))
        ok = True
        for k, rubric in enumerate(ITEMS):
            for i, j in learner_pairs:
                p = resolve(pid, rubric, "learner", i, "learner", j)
                if p is None:
                    ok = False; dropped["learner_pair_unresolved"] += 1; break
                A_LL[k, i, j] = p - 0.5
                A_LL[k, j, i] = 0.5 - p
            if not ok:
                break
            for i, j in cross_pairs:
                p = resolve(pid, rubric, "learner", i, "comparator", j)
                if p is None:
                    ok = False; dropped["cross_pair_unresolved"] += 1; break
                if (pool[pid]["learner"].get(i) ==
                        pool[pid]["comparator"].get(j) is not None):
                    identity_ties[pid] += 1
                A_LR[k, i, j] = p - 0.5
            if not ok:
                break
            # the reference triangle is now required: d = V_beta(mu) is defined
            # on it, so a prompt without it has no disagreement point and is
            # dropped rather than carried with a silently wrong d
            for i, j in learner_pairs:
                p = resolve(pid, rubric, "comparator", i, "comparator", j)
                if p is None:
                    ok = False; dropped["reference_pair_unresolved"] += 1; break
                A_RR[k, i, j] = p - 0.5
                A_RR[k, j, i] = 0.5 - p
            if not ok:
                break
        if not ok:
            continue
        idx = np.arange(POOL)
        A_LL[:, idx, idx] = 0.0
        A_RR[:, idx, idx] = 0.0
        for name, M in (("learner", A_LL), ("reference", A_RR)):
            if np.abs(M + np.swapaxes(M, -1, -2)).max() > 1e-12:
                raise SystemExit("%s payoff not antisymmetric for %s" % (name, pid))
        if max(np.abs(A_LL).max(), np.abs(A_LR).max(),
               np.abs(A_RR).max()) > 0.5 + 1e-12:
            raise SystemExit("payoff outside [-0.5, 0.5] for %s" % pid)
        # the same scalar the old file reported, so the two rounds stay comparable
        iu = np.triu_indices(POOL, 1)
        refdis[pid] = float(np.mean(np.abs(A_RR[:, iu[0], iu[1]])))
        tensors[pid] = (A_LL, A_LR, A_RR)

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
        A_LL = np.stack([tensors[p][0] for p in chunk], axis=1)
        A_LR = np.stack([tensors[p][1] for p in chunk], axis=1)
        A_RR = np.stack([tensors[p][2] for p in chunk], axis=1)
        d = out / ("shard%d" % s)
        d.mkdir(parents=True, exist_ok=True)
        path = d / "chunk0000.npz"
        # A_policy and A_ref are the names the solver reads. They are aliases of
        # A_LR and A_RR, written explicitly so an existing reader gets the
        # paper's game without being changed, while A_LL travels under its own
        # name for the pair-label consumers.
        np.savez(path, prompt_ids=np.array(chunk),
                 A_policy=A_LR, A_ref=A_RR,
                 A_LL=A_LL, A_LR=A_LR, A_RR=A_RR)
        (d / "chunk0000.manifest.json").write_text(json.dumps(
            {"prompts": len(chunk), "sha256": file_hash(path),
             "shapes": {"A_policy": list(A_LR.shape), "A_ref": list(A_RR.shape),
                        "A_LL": list(A_LL.shape), "A_LR": list(A_LR.shape),
                        "A_RR": list(A_RR.shape)},
             "roles": {"A_policy": "A_LR (learner vs reference)",
                       "A_ref": "A_RR (reference vs reference)"}},
            indent=1) + "\n")
        (d / ("complete_shard%d.json" % s)).write_text(json.dumps(
            {"shard": s, "prompts": len(chunk), "gpm_teacher": teacher,
             "bt_teacher": ("absent: the union contract uses direct order-balanced PSC "
                            "probabilities for these rows and forbids a scalar BT "
                            "projection, so none was fitted and none is written"),
             "reference_construction": ("independent reference bank: eight reference "
                                        "occurrences per prompt, so A_policy is the "
                                        "learner-by-reference cross block and A_ref is "
                                        "the reference triangle, not a copy of it")},
            indent=1) + "\n")
        written.append({"shard": s, "prompts": len(chunk),
                        "sha256": file_hash(path),
                        "shapes": {"A_policy": list(A_LR.shape),
                                   "A_ref": list(A_RR.shape),
                                   "A_LL": list(A_LL.shape)}})

    verdicts = sum(status.values())
    report = {
        "out": args.out, "judgment_tag": args.tag, "pool": args.pool,
        "construction": {
            "A_policy": ("A_LR: the 8x8 learner-by-reference cross block; NOT "
                         "antisymmetric and its diagonal need not be zero"),
            "A_ref": ("A_RR: the reference triangle, antisymmetric with zero diagonal; "
                      "this is the tensor d = V_beta(mu) is defined on"),
            "A_LL": ("the learner triangle, antisymmetric with zero diagonal; written "
                     "for single-objective pair labels, not fed to the NBPO game"),
            "corrected": ("an earlier version of this scorer set A_policy to the learner "
                          "triangle and A_ref to the cross block, which flips the sign of "
                          "the finite-pool surplus and is not the game Section 5.2 "
                          "defines; every tensor now travels under its own name"),
            "reference_triangle": "required for a prompt to be retained, not a diagnostic",
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
