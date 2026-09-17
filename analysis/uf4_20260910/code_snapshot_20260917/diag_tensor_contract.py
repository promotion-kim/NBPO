"""Verify the teacher tensor contract by response identity, not by index.

The manuscript asks for three things that index arithmetic cannot answer:

* actual self-pairs. A learner occurrence and a comparator occurrence sometimes
  carry the identical response text, because both pools are sampled at
  temperature 1 from the same policy. Those entries compare a response with
  itself and should score 0 on the margin scale.

* A(y,z) = -A(z,y) on the same response identities. The cross block stores only
  one direction, but when a learner text also occurs in the comparator pool the
  reference matrix holds that same text against the other comparators in both
  directions, so the reverse value exists and can be compared.

* whether the teacher is a function of the response texts at all: the same
  ordered pair of texts appearing in two different blocks must receive the same
  score.

Nothing is symmetrized. Counts are reported whether or not they are favourable.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/work/uf4_20260910")
DIAG = ROOT / "analysis/diag_20260914"
SPLIT = "train"
TARGET = "nash_v1"


def pool_hashes(pool):
    """prompt_id -> {role: {sample_index: response_sha256}}"""
    out = defaultdict(lambda: defaultdict(dict))
    for path in sorted(glob.glob(str(ROOT / "pools" / pool / "shard*/chunk*.jsonl"))):
        with open(path) as stream:
            for line in stream:
                r = json.loads(line)
                out[r["prompt_id"]][r["role"]][r["sample_index"]] = r["response_sha256"]
    return out


def main():
    meta = json.loads((ROOT / "targets" / TARGET / SPLIT / "tensor/meta.json").read_text())
    ids = meta["prompt_ids"]
    A = np.load(ROOT / "targets" / TARGET / SPLIT / "tensor/tensor_policy.npz")["A"]
    R = np.load(ROOT / "targets" / TARGET / SPLIT / "tensor/tensor_ref.npz")["A"]

    pool = pool_hashes("v1" if (ROOT / "pools/v1").exists() else "dev_v1")
    index = {pid: k for k, pid in enumerate(ids)}
    covered = [pid for pid in ids if pid in pool]

    self_values, cross_dup, reverse_pairs = [], [], []
    prompts_with_duplicate = 0
    for pid in covered:
        k = index[pid]
        learner = pool[pid].get("learner", {})
        comparator = pool[pid].get("comparator", {})
        if not learner or not comparator:
            continue
        by_hash = defaultdict(list)
        for j, h in comparator.items():
            by_hash[h].append(j)
        hit = False
        for i, h in learner.items():
            for j in by_hash.get(h, []):
                hit = True
                # an actual self-pair: identical text on both sides
                self_values.append(float(A[:, k, i, j].mean()))
                cross_dup.append([float(x) for x in A[:, k, i, j]])
                # the same text against every other comparator, both directions
                for j2 in comparator:
                    if j2 == j:
                        continue
                    forward = A[:, k, i, j2]        # learner text vs comparator j2
                    reverse = R[:, k, j, j2]        # same text (as comparator j) vs j2
                    reverse_pairs.append(float(np.abs(forward - reverse).max()))
        prompts_with_duplicate += int(hit)

    payload = {
        "target_set": TARGET, "split": SPLIT,
        "prompts_in_tensor": len(ids), "prompts_with_pool_cache": len(covered),
        "prompts_with_a_learner_text_repeated_in_the_comparator_pool": prompts_with_duplicate,
        "actual_self_pairs": {
            "count": len(self_values),
            "mean_margin": float(np.mean(self_values)) if self_values else None,
            "max_abs_margin": float(np.max(np.abs(self_values))) if self_values else None,
            "fraction_within_0.01": (float(np.mean(np.abs(self_values) < 0.01))
                                     if self_values else None),
            "reading": "identical text on both sides; the margin scale puts a tie at 0",
        },
        "same_identity_consistency": {
            "comparisons": len(reverse_pairs),
            "max_abs_difference": float(np.max(reverse_pairs)) if reverse_pairs else None,
            "mean_abs_difference": float(np.mean(reverse_pairs)) if reverse_pairs else None,
            "reading": ("the same ordered pair of texts scored once in the cross block and once "
                        "in the reference block; a nonzero difference means the teacher score "
                        "depends on which block it was evaluated in"),
        },
        "index_transposition_is_not_reversal": {
            "mean_abs_A_plus_AT": float(np.abs(A + np.swapaxes(A, -1, -2)).mean()),
            "mean_abs_diagonal": float(np.abs(np.diagonal(A, axis1=-2, axis2=-1)).mean()),
            "reference_block_mean_abs_R_plus_RT": float(np.abs(R + np.swapaxes(R, -1, -2)).mean()),
            "reference_block_mean_abs_diagonal": float(
                np.abs(np.diagonal(R, axis1=-2, axis2=-1)).mean()),
        },
    }
    (DIAG / "tensor_contract.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
