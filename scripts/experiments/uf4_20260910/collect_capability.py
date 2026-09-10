"""Collect every completed lm-eval capability result with its provenance."""
import json, glob
from pathlib import Path
out = {}
for p in sorted(glob.glob("/work/uf4_20260910/evaluation/capability/*/*/results.json")):
    d = json.loads(Path(p).read_text())
    arm, task = d["arm"], d["task"]
    m = d["results"].get(task, {})
    row = {k: round(v, 4) for k, v in m.items() if isinstance(v, (int, float))}
    n = (d.get("n_samples") or {}).get(task, {})
    out.setdefault(arm, {})[task] = {"num_fewshot": d["num_fewshot"], "metrics": row,
                                     "n": n.get("effective"), "harness": d["harness"],
                                     "seconds": round(d["seconds"], 1)}
for arm in sorted(out):
    print("==", arm)
    for task in sorted(out[arm]):
        r = out[arm][task]
        keys = [k for k in ("acc,none", "acc_norm,none") if k in r["metrics"]]
        vals = " / ".join("%.4f" % r["metrics"][k] for k in keys)
        errs = " / ".join("%.4f" % r["metrics"][k.replace(",none", "_stderr,none")]
                          for k in keys if k.replace(",none", "_stderr,none") in r["metrics"])
        print("   %-14s %d-shot  %s  (stderr %s)  n=%s" % (task, r["num_fewshot"], vals, errs, r["n"]))
Path("/work/uf4_20260910/analysis").mkdir(exist_ok=True)
Path("/work/uf4_20260910/analysis/capability_collected.json").write_text(json.dumps(out, indent=2) + "\n")
