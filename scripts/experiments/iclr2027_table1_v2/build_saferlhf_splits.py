#!/usr/bin/env python3
"""Immutable prompt-disjoint SafeRLHF splits, and what the graph actually contains.

PKU-SafeRLHF is the one released source in this project with *genuine
two-dimensional pairwise human annotations*: ``better_response_id`` and
``safer_response_id`` are independent judgements on the same pair, so a
conflict between helpfulness and harmlessness is observed rather than induced.
That is why it, and not UltraFeedback, can carry a nontransitivity claim --
UltraFeedback's per-completion ordinal ratings induce a total order per
objective and are therefore transitive by construction.

Splitting is by **prompt**, never by row: the same prompt appears in a mean of
1.9 rows and up to 95, so a row-level split leaks a prompt's responses across
sides. Duplicated prompts are kept -- they are different comparisons, not
duplicate labels -- and they all travel together into one split.

The report is deliberately about the *comparison graph*, because that is what
decides whether any nontransitivity claim is even testable here:

* unique responses per prompt (by the released ``response_*_sha256``);
* prompts with at least three distinct responses -- fewer than three cannot hold
  a cycle at all;
* connectivity of each prompt's comparison graph;
* **directly observable three-cycles**: triples whose three pairs are all
  annotated and whose directed edges close. These are human-observed, and are
  reported separately from anything a model predicts;
* direct disagreements: the same unordered pair annotated both ways.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import re
import unicodedata
from pathlib import Path

_WS = re.compile(r"\s+")


def norm(t) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFKC", str(t))).strip()


def sha(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


OBJECTIVES = {"helpfulness": "better_response_id", "harmlessness": "safer_response_id"}


def load_rows(cache_dir):
    from datasets import load_dataset
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train", cache_dir=cache_dir)
    rows = []
    for i, r in enumerate(ds):
        p = norm(r["prompt"])
        r0, r1 = norm(r["response_0"]), norm(r["response_1"])
        h0 = r.get("response_0_sha256") or sha(r0)
        h1 = r.get("response_1_sha256") or sha(r1)
        rows.append({
            "row_index": i,
            "prompt": p, "prompt_sha256": sha(p),
            "response_0": r0, "response_1": r1,
            "response_0_sha256": h0, "response_1_sha256": h1,
            "better_response_id": int(r["better_response_id"]),
            "safer_response_id": int(r["safer_response_id"]),
            "is_response_0_safe": bool(r["is_response_0_safe"]),
            "is_response_1_safe": bool(r["is_response_1_safe"]),
            "prompt_source": r.get("prompt_source"),
        })
    return rows


def split_prompts(prompt_hashes, seed, fractions=(0.80, 0.10, 0.10)):
    """Deterministic prompt-level assignment by a salted hash, not by shuffling.

    A hash-based assignment is reproducible without carrying an RNG state and is
    stable if the dataset grows: an existing prompt keeps its side.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    lo, hi = fractions[0], fractions[0] + fractions[1]
    out = {}
    for ph in prompt_hashes:
        u = int(sha(f"saferlhf-split-{seed}-{ph}")[:16], 16) / float(1 << 64)
        out[ph] = "train" if u < lo else ("validation" if u < hi else "test")
    return out


def graph_statistics(rows):
    """Per-objective comparison-graph statistics, computed per prompt."""
    by_prompt = collections.defaultdict(list)
    for r in rows:
        by_prompt[r["prompt_sha256"]].append(r)

    resp_counts, n_ge3, connected, singletons = [], 0, 0, 0
    stats = {o: {"observed_triples": 0, "three_cycles": 0,
                 "prompts_with_a_cycle": 0, "direct_disagreements": 0,
                 "distinct_ordered_pairs": 0}
             for o in OBJECTIVES}

    for ph, rs in by_prompt.items():
        nodes = set()
        for r in rs:
            nodes.add(r["response_0_sha256"])
            nodes.add(r["response_1_sha256"])
        resp_counts.append(len(nodes))
        if len(nodes) >= 3:
            n_ge3 += 1
        if len(nodes) == 1:
            singletons += 1

        # undirected connectivity of the comparison graph
        parent = {n: n for n in nodes}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for r in rs:
            a, b = find(r["response_0_sha256"]), find(r["response_1_sha256"])
            if a != b:
                parent[a] = b
        if len({find(n) for n in nodes}) == 1:
            connected += 1

        for obj, field in OBJECTIVES.items():
            wins = collections.Counter()       # (winner, loser) -> count
            for r in rs:
                a, b = r["response_0_sha256"], r["response_1_sha256"]
                if a == b:
                    continue                   # a self-pair carries no direction
                win, lose = (b, a) if r[field] == 1 else (a, b)
                wins[(win, lose)] += 1
            stats[obj]["distinct_ordered_pairs"] += len(wins)
            und = collections.defaultdict(set)
            for (w, l) in wins:
                und[frozenset((w, l))].add((w, l))
            stats[obj]["direct_disagreements"] += sum(1 for v in und.values() if len(v) > 1)

            # majority-resolved direction per unordered pair, then look for cycles
            edge = {}
            for pair, dirs in und.items():
                if len(dirs) > 1:
                    a, b = tuple(pair)
                    ca, cb = wins.get((a, b), 0), wins.get((b, a), 0)
                    if ca == cb:
                        continue               # a genuine split: no direction to use
                    edge[pair] = (a, b) if ca > cb else (b, a)
                else:
                    edge[pair] = next(iter(dirs))
            nl = sorted(nodes)
            if len(nl) < 3:
                continue
            found = False
            for a, b, c in itertools.combinations(nl, 3):
                pairs = [frozenset((a, b)), frozenset((b, c)), frozenset((a, c))]
                if not all(p in edge for p in pairs):
                    continue
                stats[obj]["observed_triples"] += 1
                succ = collections.defaultdict(set)
                for p in pairs:
                    w, l = edge[p]
                    succ[w].add(l)
                # a 3-cycle on three nodes: every node has out-degree exactly 1
                if all(len(succ[n]) == 1 for n in (a, b, c)):
                    stats[obj]["three_cycles"] += 1
                    found = True
            if found:
                stats[obj]["prompts_with_a_cycle"] += 1

    n = len(resp_counts)
    return {
        "n_prompts": n,
        "n_rows": len(rows),
        "unique_responses_per_prompt": {
            "min": min(resp_counts), "max": max(resp_counts),
            "mean": sum(resp_counts) / n,
            "prompts_with_one_response_pair_only": singletons},
        "prompts_with_at_least_three_responses": n_ge3,
        "prompts_with_a_connected_comparison_graph": connected,
        "per_objective": {
            o: {**v,
                "three_cycle_rate_over_observed_triples":
                    (v["three_cycles"] / v["observed_triples"]) if v["observed_triples"] else None}
            for o, v in stats.items()},
    }


def label_balance(rows):
    out = {}
    for obj, field in OBJECTIVES.items():
        c = collections.Counter(r[field] for r in rows)
        tot = sum(c.values())
        out[obj] = {"response_0_preferred": c[0], "response_1_preferred": c[1],
                    "fraction_response_1": c[1] / tot if tot else None}
    # the pair-level conflict rate is the reason this dataset is here at all
    conflict = sum(1 for r in rows
                   if r["better_response_id"] != r["safer_response_id"])
    out["helpfulness_harmlessness_conflict_rate"] = conflict / len(rows)
    out["n_conflicting_rows"] = conflict
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--fractions", type=float, nargs=3, default=(0.80, 0.10, 0.10))
    args = ap.parse_args()

    rows = load_rows(args.cache_dir)
    prompts = sorted({r["prompt_sha256"] for r in rows})
    assign = split_prompts(prompts, args.seed, tuple(args.fractions))
    for r in rows:
        r["split"] = assign[r["prompt_sha256"]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": "PKU-Alignment/PKU-SafeRLHF", "split": "train",
                "seed": args.seed, "fractions": list(args.fractions),
                "assignment": "sha256(f'saferlhf-split-{seed}-{prompt_sha256}') -> [0,1)",
                "objectives": OBJECTIVES,
                "duplicate_policy": ("rows sharing a prompt are distinct comparisons and "
                                     "are all kept; they travel together into one split"),
                "tie_policy": ("the released fields are binary ids with no tie encoding, "
                               "so no tie is invented"),
                "total_rows": len(rows), "total_prompts": len(prompts),
                "splits": {}}

    per_split = collections.defaultdict(list)
    for r in rows:
        per_split[r["split"]].append(r)

    for name in ("train", "validation", "test"):
        rs = per_split[name]
        keys = ("row_index", "prompt", "prompt_sha256", "response_0", "response_1",
                "response_0_sha256", "response_1_sha256", "better_response_id",
                "safer_response_id", "is_response_0_safe", "is_response_1_safe",
                "prompt_source", "split")
        path = args.out_dir / f"saferlhf_{name}.jsonl"
        path.write_text("\n".join(
            json.dumps({k: r[k] for k in keys}) for r in rs) + "\n")
        phs = sorted({r["prompt_sha256"] for r in rs})
        manifest["splits"][name] = {
            "path": str(path),
            "n_rows": len(rs), "n_prompts": len(phs),
            "prompt_set_sha256": sha("\n".join(phs)),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "label_balance": label_balance(rs),
            "graph": graph_statistics(rs),
        }
        print(f"[{name}] {len(rs)} rows / {len(phs)} prompts", flush=True)

    sets = {n: set(json.loads(l)["prompt_sha256"]
                   for l in (args.out_dir / f"saferlhf_{n}.jsonl").read_text().splitlines()
                   if l.strip())
            for n in ("train", "validation", "test")}
    overlaps = {}
    for a, b in itertools.combinations(sets, 2):
        inter = sets[a] & sets[b]
        overlaps[f"{a}|{b}"] = len(inter)
        if inter:
            raise SystemExit(f"prompt overlap between {a} and {b}: {len(inter)} prompts")
    manifest["pairwise_prompt_overlap"] = overlaps
    manifest["disjointness_verified_from_written_files"] = True

    (args.out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "splits"}, indent=2))
    for n, v in manifest["splits"].items():
        print(f"\n=== {n} ===")
        print(json.dumps({kk: vv for kk, vv in v.items()
                          if kk in ("n_rows", "n_prompts", "label_balance", "graph")},
                         indent=2))


if __name__ == "__main__":
    main()
