#!/usr/bin/env python3
"""Split the held-out prompts into a validation half and a test half.

The pair builder's ``--test-prompts`` gives a two-way split, but tuning needs a
third set: every hyperparameter decision is made on VALIDATION, and the test
half is opened once, for the gate. Assigning the halves here -- by a salted hash
of the prompt id, not by file order -- means the assignment is reproducible from
the salt alone and cannot drift with row order.

The split is written as a prompt_id -> half map rather than as two files, so the
single precomputed held-out split serves both and there is no second GPU pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, required=True,
                    help="pairs_test.jsonl from the builder: the held-out prompts")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--salt", default="nbpo-smoke-valtest")
    ap.add_argument("--validation-fraction", type=float, default=0.5)
    args = ap.parse_args()

    ids = sorted({json.loads(l)["prompt_id"] for l in args.pairs.open() if l.strip()})
    order = sorted(ids, key=lambda p: hashlib.sha256(f"{args.salt}|{p}".encode()).hexdigest())
    n_val = round(len(order) * args.validation_fraction)
    val, test = set(order[:n_val]), set(order[n_val:])
    out = {
        "salt": args.salt,
        "source_pairs": str(args.pairs),
        "source_pairs_sha256": hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
        "n_prompts": len(ids),
        "n_validation": len(val),
        "n_test": len(test),
        "assignment": {p: ("validation" if p in val else "test") for p in ids},
        "protocol": ("validation selects eta, learning rate and steps; the test half "
                     "is scored once, for the gate, and never used for selection"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "assignment"}, indent=2))


if __name__ == "__main__":
    main()
