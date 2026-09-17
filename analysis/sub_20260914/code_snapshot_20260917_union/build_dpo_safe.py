"""Scalarized-DPO pairs for the Safe panel, labelled by a finite BT projection.

The contract for this campaign puts the baseline's chosen/rejected pairs on the
shared learner-comparator comparisons, so a row here contrasts one learner
occurrence with one comparator occurrence -- 64 pairs per prompt -- rather than
two learners. That keeps the baseline inside the same 92-pair feedback budget
every other arm reads: no learner-learner judgment exists in this campaign, and
none is invented here.

The label comes from the per-prompt BT projection the scorer already fitted on
those same comparisons:

    r_k standardized by the TRAIN pool's mean and sd over all occurrences
    r_w = 1/2 r_helpfulness + 1/2 r_harmlessness
    chosen = the side with the larger r_w, and the pair is dropped when
             |gap| < the declared tie threshold

Calibration is fitted on train only and written once; dev reuses it. This is a
finite BT projection on 16 nodes per prompt, not a globally trained reward
model, and the artifact says so.

Rows carry exactly the columns materialize_dpo_dataset.py requires, including
the chat-template hash of the tokenizer the training run will load, so the
dataset cannot be built against a different template than the pool was sampled
with.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

UF = Path("/work/uf4_20260910")
OBJECTIVES = ("helpfulness", "harmlessness")
WEIGHTS = {"safe_uniform": [0.5, 0.5]}


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_bt(score_root, shards):
    """prompt_id -> r_bt[K, 2, POOL], hash-verified like the solver does."""
    out = {}
    for shard in range(shards):
        d = Path(score_root) / ("shard%d" % shard)
        for path in sorted(d.glob("chunk*.npz")):
            meta = json.loads((d / (path.stem + ".manifest.json")).read_text())
            if file_hash(path) != meta["sha256"]:
                raise ValueError("score chunk hash mismatch: %s" % path)
            a = np.load(path, allow_pickle=True)
            for i, pid in enumerate([str(x) for x in a["prompt_ids"]]):
                out[pid] = a["r_bt"][:, i]
    return out


def load_pool(pool_root, shards):
    pool, settings = {}, None
    for shard in range(shards):
        d = Path(pool_root) / ("shard%d" % shard)
        s = json.loads((d / "settings.json").read_text())
        comparable = {k: v for k, v in s.items() if k != "shard"}
        if settings is None:
            settings = comparable
        elif settings != comparable:
            raise ValueError("pool shards disagree on their settings")
        for path in sorted(d.glob("chunk*.jsonl")):
            manifest = json.loads((d / (path.stem + ".manifest.json")).read_text())
            if file_hash(path) != manifest["sha256"]:
                raise ValueError("pool chunk hash mismatch: %s" % path)
            with path.open() as stream:
                for line in stream:
                    e = json.loads(line)
                    pool.setdefault(e["prompt_id"], {}).setdefault(
                        e["role"], {})[e["sample_index"]] = e
    return pool, settings


def calibration(bt, name="safe_uniform"):
    """Train-pool mean and sd per objective, over every sampled occurrence."""
    stacked = np.stack([v for _pid, v in sorted(bt.items())], axis=1)   # (K, X, 2, I)
    flat = stacked.reshape(stacked.shape[0], -1)
    mean = flat.mean(axis=1)
    sd = flat.std(axis=1, ddof=1)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return {"objectives": list(OBJECTIVES),
            # materialize_dpo_dataset.py reads weights_order and
            # tie_threshold_standardized by those names, so they are written
            # under them rather than under synonyms.
            "weights_order": list(OBJECTIVES),
            "weights": WEIGHTS,
            "standardization": {"mean": mean.tolist(), "sd": sd.tolist(),
                                "fitted_on": "train pool, all learner and comparator "
                                             "occurrences",
                                "refitted_on_dev": False},
            "label_source": ("per-prompt finite Bradley-Terry projection on the shared "
                             "learner-comparator and comparator-comparator comparisons; "
                             "not a globally trained reward model"),
            "pairs": "learner i against comparator j, 64 per prompt",
            "declared_before_outcomes": True,
            "tie_threshold_standardized": None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, choices=("train", "dev"))
    ap.add_argument("--pool", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--weight", default="safe_uniform")
    ap.add_argument("--tie-threshold", type=float, default=0.05,
                    help="on the standardized scalarized gap, declared before outcomes")
    ap.add_argument("--out-root", default=str(UF / "dpo/safe_v1/pairs"))
    ap.add_argument("--calibration", default=str(UF / "dpo/safe_v1/calibration.json"))
    ap.add_argument("--base-model", default="/work/models/bases/Llama-3.1-8B-Instruct")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from mnpo_scripts.precompute_provenance import tokenizer_content_hashes

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    template_hash = tokenizer_content_hashes(tokenizer)["chat_template_hash"]

    pool, settings = load_pool(UF / "pools" / args.pool, args.shards)
    bt = load_bt(UF / "scores" / args.scores, args.shards)
    revision = settings["model_revision"]

    cal_path = Path(args.calibration)
    if args.split == "train":
        cal = calibration(bt, args.weight)
        cal["tie_threshold_standardized"] = args.tie_threshold
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        if not cal_path.exists():
            cal_path.write_text(json.dumps(cal, indent=2) + "\n")
    cal = json.loads(cal_path.read_text())
    mean = np.asarray(cal["standardization"]["mean"])
    sd = np.asarray(cal["standardization"]["sd"])
    w = np.asarray(cal["weights"][args.weight])

    out_dir = Path(args.out_root) / args.weight
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / ("%s.jsonl" % args.split)
    if dest.exists():
        print(json.dumps({"skipped": "already built", "path": str(dest)}))
        return 0

    kept = tied = flipped = 0
    prompts = sorted(set(pool) & set(bt))
    started = time.time()
    tmp = dest.with_suffix(".jsonl.tmp")
    with tmp.open("w") as stream:
        for pid in prompts:
            learners, comparators = pool[pid]["learner"], pool[pid]["comparator"]
            r = bt[pid]                                     # (K, 2, I)
            std = (r - mean[:, None, None]) / sd[:, None, None]
            scal = np.tensordot(w, std, axes=(0, 0))        # (2, I) roles x occurrences
            for i in sorted(learners):
                for j in sorted(comparators):
                    gap = float(scal[0, i] - scal[1, j])
                    if abs(gap) < args.tie_threshold:
                        tied += 1
                        continue
                    a, b = learners[i], comparators[j]
                    if gap < 0:
                        a, b = b, a
                        flipped += 1
                    kept += 1
                    row = {
                        "prompt_id": pid,
                        "prompt": a["prompt"],
                        "prompt_input_ids": a["prompt_token_ids"],
                        "prompt_attention_mask": [1] * len(a["prompt_token_ids"]),
                        "chosen": a["response"],
                        "chosen_input_ids": a["input_ids"],
                        "chosen_attention_mask": a["attention_mask"],
                        "chosen_labels": a["labels"],
                        "chosen_candidate_id": a["candidate_id"],
                        "rejected": b["response"],
                        "rejected_input_ids": b["input_ids"],
                        "rejected_attention_mask": b["attention_mask"],
                        "rejected_labels": b["labels"],
                        "rejected_candidate_id": b["candidate_id"],
                        "model_revision": revision,
                        "chat_template_hash": template_hash,
                        "dpo_weight_name": args.weight,
                        "dpo_weight_vector": [float(x) for x in w],
                        "dpo_scalarized_gap": abs(gap),
                        "dpo_tie_threshold_standardized": args.tie_threshold,
                        "dpo_label_source": ("finite BT projection on the shared "
                                             "learner-comparator comparisons, "
                                             "train-pool standardization"),
                        "dpo_pair_kind": "learner_vs_comparator",
                    }
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(dest)

    seen = kept + tied
    report = {"split": args.split, "weight": args.weight, "prompts": len(prompts),
              "pairs_considered": seen, "kept": kept, "tied": tied, "flipped": flipped,
              "usable_fraction": kept / max(seen, 1),
              "tie_threshold": args.tie_threshold,
              "pair_kind": "learner_vs_comparator (64 per prompt)",
              "seconds": time.time() - started,
              "path": str(dest), "sha256": file_hash(dest),
              "bytes": dest.stat().st_size,
              "calibration": str(cal_path), "calibration_sha256": file_hash(cal_path),
              "builder_sha256": file_hash(__file__)}
    (out_dir / ("build_report_%s.json" % args.split)).write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("split", "prompts", "kept", "tied", "flipped",
                       "usable_fraction", "bytes")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
