"""Build MOPO's training rows: one per (prompt, candidate), carrying rho(y).

Why not reuse the shared 28-pair rows. MOPO's loss is -rho(y) log pi(y) on the
chosen side only, and the shared enumeration is combinations(range(8), 2) with
chosen = i < j, so candidate 0 is chosen seven times and candidate 7 never.
Feeding those rows to MOPO would weight the behaviour-cloning estimate by a
candidate's index instead of by its rho, which is not MOPO.

Why no tokenizer call. Every candidate's token fields already exist in the
canonical pair file -- each candidate appears in seven of the 28 pairs, on one
side or the other -- and the canonical validator that file already passed
enforces that a candidate's tokens do not depend on its pair partner. So the
tokens are harvested rather than recomputed, which makes them bit-identical to
the ones the NBPO, fixed-reference, utilitarian and maxmin arms trained on. A
fresh tokenization could only differ, and any difference would be a confound in
the one comparison this arm exists to make.

The rejected side carries real tokens because the shared collator needs a pair;
it enters no term of the MOPO loss, and it is the next candidate cyclically so
the choice is fixed rather than data-dependent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/work/uf4_20260910")
SIDE_KEYS = ("input_ids", "attention_mask", "labels")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def harvest(pairs_path, prompt_ids):
    """prompt_id -> {candidate_id: {token fields}}, plus the prompt text and ids."""
    want = set(prompt_ids)
    out, prompts, carried = {}, {}, {}
    with open(pairs_path) as stream:
        for line in stream:
            row = json.loads(line)
            pid = str(row["prompt_id"])
            if pid not in want:
                continue
            prompts.setdefault(pid, {
                "prompt": row["prompt"],
                "prompt_input_ids": row["prompt_input_ids"],
                "prompt_attention_mask": row["prompt_attention_mask"]})
            carried.setdefault(pid, {k: row[k] for k in
                                     ("chat_template_hash", "model_revision",
                                      "pool_settings_sha256", "teacher_manifest_sha256",
                                      "candidate_token_schema", "split")
                                     if k in row})
            group = out.setdefault(pid, {})
            for side in ("chosen", "rejected"):
                # Key on the candidate INDEX, not the response id. The ids are
                # "<prompt_id>:learner:<i>", so they are prompt-scoped strings,
                # while rho's second axis is the candidate index of the teacher
                # tensor. The index is the thing the two artifacts share.
                cid = int(row[f"{side}_candidate_index"])
                fields = {k: row[f"{side}_{k}"] for k in SIDE_KEYS}
                fields["text"] = row[side]
                fields["token_sha256"] = row.get(f"{side}_token_sha256")
                fields["text_sha256"] = row.get(f"{side}_text_sha256")
                fields["response_id"] = str(row[f"{side}_response_id"])
                old = group.setdefault(cid, fields)
                if old["token_sha256"] != fields["token_sha256"]:
                    raise ValueError("candidate %s tokens differ between pairs of prompt %s"
                                     % (cid, pid))
    return out, prompts, carried


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rho-from", default="mopo_v1")
    ap.add_argument("--pairs-from", default="nash_v1")
    ap.add_argument("--out-name", default="mopo_v1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rho_root = ROOT / "targets" / args.rho_from
    pair_root = ROOT / "targets" / args.pairs_from
    solver_hash = file_hash(pair_root / "train/solver/solution.json")
    report = {"rho_from": str(rho_root), "pairs_from": str(pair_root),
              "solver_artifact_sha256": solver_hash, "splits": {}}

    for split in ("train", "dev"):
        meta = json.loads((pair_root / split / "tensor/meta.json").read_text())
        prompt_ids = [str(x) for x in meta["prompt_ids"]]
        learner_ids = list(meta["policy_learner_ids"])
        rho = np.load(rho_root / split / "rho.npz")["rho"]
        if rho.shape != (len(prompt_ids), len(learner_ids)):
            raise ValueError("rho shape %s does not match %d prompts x %d candidates"
                             % (rho.shape, len(prompt_ids), len(learner_ids)))
        dev = float(np.abs(rho.mean(axis=1) - 1.0).max())
        if dev > 1e-9:
            raise ValueError("rho is not an importance ratio: max |mean-1| = %.3e" % dev)

        pairs_path = pair_root / "pairs" / f"{split}.jsonl"
        groups, prompts, carried = harvest(pairs_path, prompt_ids)
        missing = [p for p in prompt_ids if len(groups.get(p, {})) != len(learner_ids)]
        if missing:
            raise ValueError("%d prompts did not yield all %d candidates, e.g. %s"
                             % (len(missing), len(learner_ids), missing[:3]))

        out_path = ROOT / "targets" / args.out_name / "pairs" / f"{split}.jsonl"
        rows_written = 0
        if not args.dry_run:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sink = out_path.open("w")
        for x, pid in enumerate(prompt_ids):
            group = groups[pid]
            for i in range(len(learner_ids)):
                partner = (i + 1) % len(learner_ids)
                chosen, rejected = group[i], group[partner]
                row = {
                    "prompt_id": pid, "prompt": prompts[pid]["prompt"],
                    "prompt_input_ids": prompts[pid]["prompt_input_ids"],
                    "prompt_attention_mask": prompts[pid]["prompt_attention_mask"],
                    "chosen": chosen["text"], "rejected": rejected["text"],
                    "chosen_response_id": chosen["response_id"],
                    "rejected_response_id": rejected["response_id"],
                    "chosen_candidate_index": i, "rejected_candidate_index": partner,
                    "chosen_text_sha256": chosen["text_sha256"],
                    "rejected_text_sha256": rejected["text_sha256"],
                    "chosen_token_sha256": chosen["token_sha256"],
                    "rejected_token_sha256": rejected["token_sha256"],
                    "ronpo_target": float(rho[x, i]),
                    "target_mode": "mopo_rho", "target_units": "importance_ratio",
                    "nbpo_num_candidates": len(learner_ids),
                    "solver_artifact_sha256": solver_hash,
                    "rejected_side_note": "carried for the collator; enters no MOPO loss term",
                    **{f"chosen_{k}": chosen[k] for k in SIDE_KEYS},
                    **{f"rejected_{k}": rejected[k] for k in SIDE_KEYS},
                    **carried[pid]}
                rows_written += 1
                if not args.dry_run:
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        if not args.dry_run:
            sink.close()
        entry = {"prompts": len(prompt_ids), "candidates": len(learner_ids),
                 "rows": rows_written, "rho_mean_max_dev": dev,
                 "rho_min": float(rho.min()), "rho_max": float(rho.max()),
                 "source_pairs_sha256": file_hash(pairs_path)}
        if not args.dry_run:
            entry["path"] = str(out_path)
            entry["sha256"] = file_hash(out_path)
        report["splits"][split] = entry
        print(json.dumps({"split": split, **entry}), flush=True)

    if args.dry_run:
        print(json.dumps({"dry_run": True, "nothing_written": True}))
        return
    out = ROOT / "targets" / args.out_name / "rows_complete.json"
    report["builder_sha256"] = file_hash(Path(__file__))
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"written": str(out)}), flush=True)


if __name__ == "__main__":
    main()
