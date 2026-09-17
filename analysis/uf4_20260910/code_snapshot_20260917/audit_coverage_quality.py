"""Audit coverage, order sensitivity, and whether the frozen GPM beats the frozen BT.

Coverage, ties, triangle rate and order gap come straight from the judgments.
The predictive comparison scores BOTH frozen teachers on the SAME complete
panel-B pairs with the mean per-judgment Brier loss (P - X)^2, paired by prompt.
Neither teacher is fitted or calibrated on the audit; both were selected on the
preference-model dev split before any audit response existed.

A negative GPM minus BT difference favours the general preference model. This is
a predictive comparison and establishes nothing about cycles.
"""
from __future__ import annotations

import argparse, hashlib, itertools, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/work/uf4_20260910/code")
from analyze_audit import load, PAIRS, DRAWS_PER_ORDER, edge_estimates, graph_from, directed_triangles

ROOT = Path("/work/uf4_20260910")
RUBRICS = ("instruction_following", "truthfulness", "honesty", "helpfulness", "joint_control")
INDIVIDUAL = RUBRICS[:4]
OBJ_INDEX = {name: i for i, name in enumerate(INDIVIDUAL)}


def file_hash(p):
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-name", default="v1")
    ap.add_argument("--split-file", default="audit_100")
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    base = ROOT / "audit" / args.audit_name
    jpath = base / "judgments" / args.split_file / "judgments.jsonl"
    data = load(jpath)
    prompts = sorted({k[1] for k in data})

    raw = defaultdict(lambda: {"scheduled": 0, "valid": 0, "tie": 0})
    with jpath.open() as stream:
        for line in stream:
            r = json.loads(line)
            raw[r["rubric"]]["scheduled"] += 1
            if r["status"] == "ok":
                raw[r["rubric"]]["valid"] += 1
                if r["value_for_i"] == 0.5:
                    raw[r["rubric"]]["tie"] += 1

    coverage = {}
    for rubric in RUBRICS:
        c = raw[rubric]
        gaps, triangles = [], 0
        for prompt in prompts:
            p_hat = edge_estimates(data, rubric, prompt, "A")
            for pair in PAIRS:
                v0 = data.get((rubric, prompt, "A", pair, 0), [])
                v1 = data.get((rubric, prompt, "A", pair, 1), [])
                if len(v0) == DRAWS_PER_ORDER and len(v1) == DRAWS_PER_ORDER:
                    gaps.append(abs(float(np.mean(v0)) - float(np.mean(v1))))
            g = graph_from(p_hat)
            if g is not None:
                triangles += len(directed_triangles(g))
        coverage[rubric] = {
            "missing_pct": 100.0 * (1 - c["valid"] / max(c["scheduled"], 1)),
            "tie_pct": 100.0 * c["tie"] / max(c["valid"], 1),
            "triangle_pct": 100.0 * triangles / (4 * len(prompts)),
            "order_gap": float(np.mean(gaps)) if gaps else None,
            "scheduled": c["scheduled"], "valid": c["valid"],
            "complete_pairs_used_for_gap": len(gaps)}

    # ---- Brier on identical complete panel-B pairs
    from transformers import AutoTokenizer
    sys.path.insert(0, str(ROOT / "code"))
    from score_uf4_pool import Serializer, load_teacher, run_batches
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = str(ROOT / "assets/ModernBERT-base")
    tok = AutoTokenizer.from_pretrained(backbone, local_files_only=True)
    ser = Serializer(tok, 8192, 2048)
    gpm, gpm_info = load_teacher(str(ROOT / "teacher/gpm_s42"), backbone, device)
    bt, bt_info = load_teacher(str(ROOT / "teacher/bt_s42"), backbone, device)

    resp = {}
    with (base / "responses" / args.split_file / "responses.jsonl").open() as stream:
        for line in stream:
            e = json.loads(line)
            resp.setdefault(e["prompt_id"], {})[e["response_index"]] = e

    # one forward per (prompt, pair); the teachers are symmetric by construction
    rows, index = [], []
    for prompt in prompts:
        for i, j in PAIRS:
            rows.append(ser.pair_rows(resp[prompt][0]["instruction"],
                                      resp[prompt][i]["response"], resp[prompt][j]["response"]))
            index.append((prompt, (i, j)))
    audit = {}
    P_gpm = torch.sigmoid(run_batches(gpm, ser.gpm, rows, device, args.micro, audit)).numpy()
    P_bt = torch.sigmoid(run_batches(bt, ser.bt, rows, device, args.micro, audit)).numpy()
    lookup = {key: n for n, key in enumerate(index)}

    quality = {}
    for rubric in INDIVIDUAL:
        k = OBJ_INDEX[rubric]
        per_prompt_gpm, per_prompt_bt, kept = defaultdict(list), defaultdict(list), 0
        for prompt in prompts:
            for pair in PAIRS:
                vals = []
                for order in (0, 1):
                    v = data.get((rubric, prompt, "B", pair, order), [])
                    if len(v) != DRAWS_PER_ORDER:
                        vals = None
                        break
                    vals += v
                if vals is None:
                    continue
                kept += 1
                n = lookup[(prompt, pair)]
                for x in vals:
                    per_prompt_gpm[prompt].append((P_gpm[n, k] - x) ** 2)
                    per_prompt_bt[prompt].append((P_bt[n, k] - x) ** 2)
        keys = sorted(per_prompt_gpm)
        g_means = np.array([np.mean(per_prompt_gpm[p]) for p in keys])
        b_means = np.array([np.mean(per_prompt_bt[p]) for p in keys])
        rng = np.random.default_rng(20260911)
        diffs = []
        for _ in range(args.bootstrap):
            pick = rng.integers(0, len(keys), len(keys))
            diffs.append(float(g_means[pick].mean() - b_means[pick].mean()))
        diffs = np.array(diffs)
        quality[rubric] = {
            "bt_brier": float(b_means.mean()), "gpm_brier": float(g_means.mean()),
            "gpm_minus_bt": float(g_means.mean() - b_means.mean()),
            "ci95": [float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975))],
            "complete_B_pairs": kept, "prompts_with_data": len(keys)}
        print(json.dumps({"rubric": rubric, **{k2: round(v, 5) if isinstance(v, float) else v
                                               for k2, v in quality[rubric].items()}}), flush=True)

    report = {"coverage": coverage, "prediction": quality,
              "gpm_teacher": gpm_info, "bt_teacher": bt_info,
              "note": ("Brier uses identical complete panel-B pairs for both teachers; neither is "
                       "fitted or calibrated on the audit. A predictive comparison, not evidence "
                       "about cycles."),
              "judgments_sha256": file_hash(jpath), "source_sha256": file_hash(__file__)}
    out = ROOT / "analysis/audit/coverage_quality.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    for r in RUBRICS:
        c = coverage[r]
        print(json.dumps({"rubric": r, "missing_pct": round(c["missing_pct"], 3),
                          "tie_pct": round(c["tie_pct"], 2),
                          "triangle_pct": round(c["triangle_pct"], 2),
                          "order_gap": round(c["order_gap"], 4)}), flush=True)
    print(json.dumps({"written": str(out)}), flush=True)


if __name__ == "__main__":
    main()
