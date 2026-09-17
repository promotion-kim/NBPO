"""Select at most one cyclic triangle per rubric from panel A, before opening B.

The manuscript's rule, verbatim: complete A judgments, all three cycle directions
agreeing across BOTH presentation orders, maximize the minimum edge margin,
ties broken by prompt and response ids. The manifest is committed here so the
selection cannot depend on anything in panel B.

Panel B is never read by this file.
"""
from __future__ import annotations

import argparse, hashlib, itertools, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/work/uf4_20260910/code")
from analyze_audit import load, PAIRS, DRAWS_PER_ORDER, directed_triangles, graph_from, edge_estimates

ROOT = Path("/work/uf4_20260910")
INDIVIDUAL = ("instruction_following", "truthfulness", "honesty", "helpfulness")


def file_hash(p):
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def order_means(data, rubric, prompt, panel, pair):
    out = {}
    for order in (0, 1):
        vals = data.get((rubric, prompt, panel, pair, order), [])
        out[order] = float(np.mean(vals)) if len(vals) == DRAWS_PER_ORDER else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-name", default="v1")
    ap.add_argument("--split-file", default="audit_100")
    args = ap.parse_args()
    base = ROOT / "audit" / args.audit_name
    jpath = base / "judgments" / args.split_file / "judgments.jsonl"
    data = load(jpath)
    prompts = sorted({k[1] for k in data})

    manifest = {"rule": ("complete A judgments; all three cycle directions agree in BOTH "
                         "presentation orders; maximize the minimum edge margin; ties broken "
                         "by prompt id then response ids"),
                "panel_B_read": False,
                "judgments_sha256": file_hash(jpath),
                "source_sha256": file_hash(__file__),
                "selected": {}}
    for rubric in INDIVIDUAL:
        best = None
        for prompt in prompts:
            p_hat = edge_estimates(data, rubric, prompt, "A")
            g = graph_from(p_hat)
            if g is None:
                continue
            for tri in sorted(directed_triangles(g)):
                margins, agree = [], True
                for (u, v) in tri:
                    pair = (u, v) if u < v else (v, u)
                    om = order_means(data, rubric, prompt, "A", pair)
                    if om[0] is None or om[1] is None:
                        agree = False
                        break
                    # direction u->v means the value for the lower index must
                    # exceed 1/2 in the u->v sense, in BOTH orders separately
                    for order in (0, 1):
                        val = om[order] if pair == (u, v) else 1.0 - om[order]
                        if not val > 0.5:
                            agree = False
                    if not agree:
                        break
                    avg = 0.5 * (om[0] + om[1])
                    margins.append(abs((avg if pair == (u, v) else 1.0 - avg) - 0.5))
                if not agree or len(margins) != 3:
                    continue
                key = (-min(margins), prompt, tuple(sorted(tri)))
                if best is None or key < best[0]:
                    best = (key, {"prompt_id": prompt, "triangle": [list(e) for e in sorted(tri)],
                                  "min_edge_margin": float(min(margins)),
                                  "edge_margins": [float(m) for m in margins]})
        manifest["selected"][rubric] = best[1] if best else {
            "prompt_id": None, "triangle": None,
            "reason": "no cyclic triangle in panel A agrees in both presentation orders"}
        print(json.dumps({"rubric": rubric, **manifest["selected"][rubric]}), flush=True)

    out = base / "witness_manifest.json"
    if out.exists():
        raise SystemExit(f"Refusing to overwrite a committed manifest: {out}")
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"written": str(out),
                      "n_selected": sum(1 for v in manifest["selected"].values()
                                        if v.get("triangle"))}), flush=True)


if __name__ == "__main__":
    main()
