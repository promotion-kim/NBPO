"""Attach order-balanced soft preference labels to a panel's pair dataset.

The scalarized-DPO and mean-preference-INPO rows of Table 3 need, for each
training pair, the probability that the chosen response is preferred to the
rejected one under the SAME judgments the NBPO teacher used. That probability is
the panel's objective-averaged direct PSC probability

    p = (1/K) sum_k [ A_LL[k, x, i, j] + 1/2 ],

where A_LL is the antisymmetric, zero-diagonal learner block written by the
panel scorer, i and j are the row's own candidate indices, and the +1/2 turns
the antisymmetric deviation back into a probability. No Bradley-Terry fit and no
surrogate reward model enters: these are the measured order-averaged verdicts.

The tokenization, prompts and pair set are copied unchanged from the NBPO-PW
dataset of the same panel, so the two baselines see exactly the pairs the NBPO
arms see. Only the two probability columns are added.

Written with the trainer's own datasets version (deps_train first on PYTHONPATH)
because a newer writer emits a feature type the trainer's reader rejects.
"""
from __future__ import annotations

import argparse, glob, hashlib, json
from pathlib import Path

import numpy as np

UF = Path("/work/uf4_20260910")


def load_policy_block(scores_dir, shards):
    """prompt_id -> A_LL, the antisymmetric learner block."""
    out = {}
    for shard in range(shards):
        d = Path(scores_dir) / ("shard%d" % shard)
        for path in sorted(d.glob("chunk*.npz")):
            z = np.load(path, allow_pickle=True)
            pids = [str(v) for v in z["prompt_ids"]]
            # A_LL, explicitly: the pair label is a learner-versus-learner
            # probability, and since the A01 fix A_policy carries the
            # learner-versus-reference cross block instead. A score directory
            # written before that fix has no A_LL, and must be re-scored rather
            # than read here, so its absence raises instead of falling back.
            if "A_LL" not in z.files:
                raise SystemExit(
                    "%s predates the A01 tensor-role fix: it has no A_LL, and its "
                    "A_policy is the learner triangle only by coincidence of the old "
                    "wiring. Re-score this panel with union_score_panel.py before "
                    "building soft labels." % path)
            A = z["A_LL"]
            for x, pid in enumerate(pids):
                out[pid] = np.asarray(A[:, x], dtype=np.float64)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", required=True, help="us1, ut1 or uw1")
    ap.add_argument("--source-dataset", required=True,
                    help="the NBPO-PW dataset of this panel, copied pair for pair")
    ap.add_argument("--scores", required=True)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from datasets import load_from_disk
    import datasets as ds_mod
    print(json.dumps({"datasets_version": ds_mod.__version__,
                      "writer_must_match_trainer": True}), flush=True)

    blocks = load_policy_block(args.scores, args.shards)
    K = None
    data = load_from_disk(args.source_dataset)
    stats = {}

    def attach(batch):
        ps = []
        for pid, i, j in zip(batch["prompt_id"], batch["chosen_candidate_index"],
                             batch["rejected_candidate_index"]):
            A = blocks[pid]
            vals = A[:, i, j] + 0.5
            ps.append(float(vals.mean()))
        return {"chosen_probs": ps, "rejected_probs": [1.0 - p for p in ps]}

    out = {}
    for split in data:
        d = data[split]
        missing = sorted({p for p in d["prompt_id"] if p not in blocks})
        if missing:
            raise SystemExit("%s: %d prompts have no score block" % (split, len(missing)))
        any_pid = d["prompt_id"][0]
        K = blocks[any_pid].shape[0]
        # the block must be antisymmetric with a zero diagonal, or +1/2 is wrong
        A = blocks[any_pid]
        if float(np.abs(A + np.swapaxes(A, 1, 2)).max()) > 1e-9:
            raise SystemExit("learner block is not antisymmetric")
        if float(np.abs(np.einsum("kii->ki", A)).max()) > 1e-12:
            raise SystemExit("learner block has a nonzero diagonal")
        d = d.map(attach, batched=True, batch_size=512)
        p = np.asarray(d["chosen_probs"], dtype=np.float64)
        q = np.asarray(d["rejected_probs"], dtype=np.float64)
        if float(np.abs(p + q - 1.0).max()) > 1e-12:
            raise SystemExit("probabilities do not sum to one")
        if p.min() < 0.0 or p.max() > 1.0:
            raise SystemExit("probability outside [0,1]: %g..%g" % (p.min(), p.max()))
        stats[split] = {"rows": len(p), "p_mean": float(p.mean()),
                        "p_sd": float(p.std()), "p_min": float(p.min()),
                        "p_max": float(p.max()),
                        "fraction_above_half": float((p > 0.5).mean())}
        out[split] = d

    from datasets import DatasetDict
    DatasetDict(out).save_to_disk(args.out)
    rec = {"panel": args.panel, "source_dataset": args.source_dataset,
           "scores": args.scores, "objectives": K,
           "label": "mean over the panel's K objectives of A_LL[k,x,i,j] + 1/2",
           "no_bt_fit": True, "no_reward_model": True,
           "datasets_version": ds_mod.__version__, "splits": stats}
    Path(args.out, "softlabel_record.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
