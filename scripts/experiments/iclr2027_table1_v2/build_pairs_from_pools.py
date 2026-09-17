#!/usr/bin/env python3
"""Turn a learner/comparator response pool into the pair list the v3 runner scores.

Emits every comparison the finite-pool tensors need, for every objective:

* **cross** pairs -- each learner response against each comparator response
  (``I x J``), which fill ``A_policy``;
* **reference** pairs -- each unordered pair among the comparators, which fill
  the skew-symmetric ``A_ref`` the disagreement point is computed from.

Only unordered reference pairs are emitted: ``A_ref`` is built with exact skew
symmetry from one judged direction per pair, so judging both directions would be
wasted budget *and* would reintroduce the asymmetry the builder exists to avoid.

Presentation order is handled downstream by the runner, which scores both.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path


def load_pool(specs):
    out = {}
    for spec in specs:
        seed, path = spec.split("=", 1)
        data = json.loads(Path(path).read_text())
        rows = data if isinstance(data, list) else data.get("responses", data)
        out[seed] = {r["prompt_id"]: r for r in rows}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--learner", nargs="+", required=True, help="seed=path.json")
    ap.add_argument("--comparator", nargs="+", required=True)
    ap.add_argument("--objectives", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-prompts", type=int, default=None)
    args = ap.parse_args()

    learner = load_pool(args.learner)
    comparator = load_pool(args.comparator)
    objectives = [o for o in args.objectives.split(",") if o]

    lseeds, cseeds = sorted(learner), sorted(comparator)
    prompts = set.intersection(*(set(v) for v in list(learner.values())
                                 + list(comparator.values())))
    missing = (set().union(*(set(v) for v in learner.values())) - prompts)
    if missing:
        raise SystemExit(f"{len(missing)} prompts are not present in every pool file; "
                         "a partial pool would silently drop cells from the tensor")
    prompt_ids = sorted(prompts)
    if args.max_prompts:
        prompt_ids = prompt_ids[:args.max_prompts]

    rows = []
    for pid in prompt_ids:
        prompt = learner[lseeds[0]][pid]["prompt"]
        for obj in objectives:
            for ls in lseeds:
                for cs in cseeds:
                    rows.append(dict(
                        pair_id=f"x|{pid}|{obj}|{ls}|{cs}", objective=obj, pool="policy",
                        prompt_id=pid, prompt=prompt,
                        learner_id=f"policy:{ls}", comparator_id=f"ref:{cs}",
                        learner_seed=ls, comparator_seed=cs,
                        response_a=learner[ls][pid]["generated_text"],
                        response_b=comparator[cs][pid]["generated_text"]))
            for a, b in itertools.combinations(cseeds, 2):
                rows.append(dict(
                    pair_id=f"r|{pid}|{obj}|{a}|{b}", objective=obj, pool="reference",
                    prompt_id=pid, prompt=prompt,
                    learner_id=f"ref:{a}", comparator_id=f"ref:{b}",
                    learner_seed=a, comparator_seed=b,
                    response_a=comparator[a][pid]["generated_text"],
                    response_b=comparator[b][pid]["generated_text"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    n_cross = sum(1 for r in rows if r["pool"] == "policy")
    print(json.dumps({
        "prompts": len(prompt_ids), "objectives": len(objectives),
        "learner_seeds": lseeds, "comparator_seeds": cseeds,
        "cross_pairs": n_cross, "reference_pairs": len(rows) - n_cross,
        "total_pairs": len(rows),
        "renderings_at_two_orders": 2 * len(rows),
        "out": str(args.out),
        "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}, indent=1))


if __name__ == "__main__":
    main()
