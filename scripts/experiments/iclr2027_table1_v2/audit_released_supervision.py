#!/usr/bin/env python3
"""Audit the released annotation schemas -- do not assume them.

The paper's terminology and the released field names have to be checked against
each other rather than trusted to match. This reports exactly what each dataset
provides so that a decision to use (or drop) a panel rests on the schema, not on
recollection.
"""
from __future__ import annotations

import argparse
import collections
import json
import unicodedata, re
from pathlib import Path

_WS = re.compile(r"\s+")
norm = lambda t: _WS.sub(" ", unicodedata.normalize("NFKC", str(t))).strip()


def audit_saferlhf(cache_dir):
    from datasets import load_dataset
    out = {"dataset": "PKU-Alignment/PKU-SafeRLHF"}
    try:
        ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train", cache_dir=cache_dir)
    except Exception as exc:
        return {**out, "status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
    out["status"] = "ok"
    out["rows"] = len(ds)
    out["fields"] = sorted(ds[0].keys())
    prompts = collections.Counter()
    resp_per_prompt = collections.defaultdict(set)
    helpf, harml, ties, missing = collections.Counter(), collections.Counter(), 0, 0
    for r in ds:
        p = norm(r.get("prompt", ""))
        prompts[p] += 1
        for k in ("response_0", "response_1"):
            if r.get(k):
                resp_per_prompt[p].add(norm(r[k])[:200])
        h = r.get("better_response_id")
        s = r.get("safer_response_id")
        if h is None or s is None:
            missing += 1
        helpf[h] += 1
        harml[s] += 1
    out["unique_prompts"] = len(prompts)
    out["prompt_duplicates"] = sum(v - 1 for v in prompts.values() if v > 1)
    n = [len(v) for v in resp_per_prompt.values()]
    out["responses_per_prompt"] = {"min": min(n), "max": max(n),
                                   "mean": sum(n) / len(n)}
    out["helpfulness_field"] = "better_response_id"
    out["harmlessness_field"] = "safer_response_id"
    out["helpfulness_class_balance"] = {str(k): v for k, v in helpf.items()}
    out["harmlessness_class_balance"] = {str(k): v for k, v in harml.items()}
    out["rows_missing_a_label"] = missing
    out["explicit_tie_encoding"] = ("none: both preference fields are binary ids, so a "
                                    "tie cannot be expressed and must not be invented")
    # prompt-disjoint split sizes at the usual proportions
    u = len(prompts)
    out["prompt_disjoint_split_plan"] = {"train": int(u * 0.8), "validation": int(u * 0.1),
                                         "test": u - int(u * 0.8) - int(u * 0.1)}
    return out


def audit_ultrafeedback(cache_dir):
    from datasets import load_dataset
    out = {"dataset": "openbmb/UltraFeedback"}
    ds = load_dataset("openbmb/UltraFeedback", split="train", cache_dir=cache_dir)
    out["status"] = "ok"
    out["rows"] = len(ds)
    out["fields"] = sorted(ds[0].keys())
    aspects = collections.Counter()
    ncomp = collections.Counter()
    scales = collections.defaultdict(collections.Counter)
    missing = collections.Counter()
    have_all4 = 0
    NEEDED = ["instruction_following", "truthfulness", "honesty", "helpfulness"]
    for r in ds:
        comps = r.get("completions") or []
        ncomp[len(comps)] += 1
        ok4 = bool(comps)
        for c in comps:
            ann = c.get("annotations") or {}
            for a in ann:
                aspects[a] += 1
            for a in NEEDED:
                v = (ann.get(a) or {}).get("Rating") if isinstance(ann.get(a), dict) else None
                if v is None:
                    missing[a] += 1
                    ok4 = False
                else:
                    scales[a][str(v)] += 1
        have_all4 += int(ok4)
    out["annotation_aspects_present"] = dict(aspects)
    out["paper_terms_vs_released_fields"] = {
        a: ("EXACT MATCH" if a in aspects else "ABSENT") for a in NEEDED}
    out["completions_per_prompt"] = dict(ncomp)
    out["rating_scales"] = {a: dict(sorted(scales[a].items())) for a in NEEDED}
    out["completions_missing_rating"] = dict(missing)
    out["prompts_with_all_four_aspects_on_every_completion"] = have_all4
    out["label_type"] = ("per-completion ordinal RATINGS (1-5), not pairwise labels; "
                         "pairwise supervision must be INDUCED from score differences")
    out["induced_label_caveat"] = (
        "score-induced pairwise labels inherit a total order per objective and are "
        "therefore transitive by construction. They must NOT be described as naturally "
        "intransitive, and any cyclicity claim needs a different source.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()
    report = {"saferlhf": audit_saferlhf(args.cache_dir),
              "ultrafeedback": audit_ultrafeedback(args.cache_dir)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)[:3000])


if __name__ == "__main__":
    main()
