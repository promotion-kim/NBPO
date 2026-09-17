#!/usr/bin/env python3
"""Export every honesty natural-control disagreement, plus a blinded review sheet.

Amendment 001 rules that UltraFeedback's per-aspect ratings are GPT-4-derived
pseudo-labels whose construct is not guaranteed to match the frozen honesty
rubric, so a disagreement between the judge and that label is *not* evidence of
a judge error until a human adjudicates it.

This writes two files:

``honesty_natural_disagreements.csv``
    the full record -- prompt, both responses, the frozen rubric text, the
    pseudo-label, the judge's verdict in each presentation order, and stable
    hashes -- so the disagreement can be re-derived and audited.

``honesty_blinded_review.csv``
    the same pairs with **both labels removed** and the A/B assignment
    re-randomised under a separate seed, so a reviewer decides from the rubric
    alone. The join key is a salted hash: without the salt file the sheet cannot
    be linked back to either label, which is what keeps the review blind.

No downstream NBPO or policy quantity is read here, and none may inform the
review.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import yaml


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controls", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--rubric", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--objective", default="honesty")
    ap.add_argument("--blind-seed", type=int, default=77003)
    args = ap.parse_args()

    rubric = yaml.safe_load(args.rubric.read_text())
    criterion = rubric["objectives"][args.objective]["criterion"].strip()
    rubric_hash = hashlib.sha256(args.rubric.read_bytes()).hexdigest()

    controls = {c["control_id"]: c for c in read_jsonl(args.controls)
                if c["objective"] == args.objective and c["family"] == "natural"}
    results = {r["pair_id"]: r for r in read_jsonl(args.results)}

    rows, blinded = [], []
    rng = random.Random(args.blind_seed)
    salt = hashlib.sha256(str(args.blind_seed).encode()).hexdigest()[:16]

    for cid, c in sorted(controls.items()):
        r = results.get(cid)
        if not r or not r.get("valid"):
            continue
        want = c["expected_verdict"]
        judged = "A" if r["delta"] > 0 else ("B" if r["delta"] < 0 else "TIE")
        if judged == want:
            continue
        obs = {o["presentation_order"]: o for o in (r.get("observations") or [])}
        rows.append({
            "control_id": cid,
            "objective": args.objective,
            "prompt": c["prompt"],
            "response_a": c["response_a"],
            "response_b": c["response_b"],
            "prompt_sha256": c["prompt_sha256"],
            "response_a_sha256": c["response_a_sha256"],
            "response_b_sha256": c["response_b_sha256"],
            "rubric_sha256": rubric_hash,
            "frozen_rubric_criterion": criterion,
            "uf_pseudo_label_preferred": want,
            "uf_rating_high": c.get("rating_high"),
            "uf_rating_low": c.get("rating_low"),
            "judge_verdict": judged,
            "judge_delta": round(r["delta"], 6),
            "judge_forward_hard_argmax": obs.get("learner_first", {}).get("hard_argmax"),
            "judge_reverse_hard_argmax": obs.get("comparator_first", {}).get("hard_argmax"),
            "judge_forward_semantic_score": round(r["semantic_score_forward"], 6),
            "judge_reverse_semantic_score": round(r["semantic_score_reverse"], 6),
            "judge_tie_probability": round(r["mean_tie_probability"], 6),
            "judge_order_gap": round(r["order_gap"], 6),
            "adjudicated": bool(r.get("adjudicated")),
        })
        # blinded: re-randomise slots under a separate seed, strip both labels
        flip = rng.random() < 0.5
        left, right = ((c["response_b"], c["response_a"]) if flip
                       else (c["response_a"], c["response_b"]))
        blinded.append({
            "review_id": hashlib.sha256((salt + cid).encode()).hexdigest()[:16],
            "objective": args.objective,
            "criterion": criterion,
            "prompt": c["prompt"],
            "response_left": left,
            "response_right": right,
            "your_verdict_LEFT_RIGHT_or_TIE": "",
            "your_confidence_1to5": "",
            "your_note": "",
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (args.out_dir / "honesty_natural_disagreements.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if blinded:
        with (args.out_dir / "honesty_blinded_review.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(blinded[0]))
            w.writeheader()
            w.writerows(blinded)
    # the salt lives apart from the sheet; without it the review cannot be unblinded
    (args.out_dir / "honesty_blind_key.json").write_text(json.dumps({
        "blind_seed": args.blind_seed, "salt": salt,
        "review_id_formula": "sha256(salt + control_id)[:16]",
        "note": ("keep this file away from whoever fills in the review sheet; it is the "
                 "only way to join a verdict back to either label"),
        "n_disagreements": len(rows)}, indent=2))
    print(json.dumps({"objective": args.objective,
                      "natural_controls_scored": len(controls),
                      "disagreements_exported": len(rows),
                      "blinded_rows": len(blinded),
                      "out_dir": str(args.out_dir)}, indent=1))


if __name__ == "__main__":
    main()
