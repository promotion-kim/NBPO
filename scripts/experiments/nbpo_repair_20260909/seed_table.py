"""Seed-level results for every arm family, with seed spread kept separate from prompt CIs."""
import json, glob, os, statistics
from pathlib import Path
import numpy as np

R = Path("/work/nbpo_repair_20260909")
FAMILIES = {
    "NBPO-MSE, 250": ["mse_short_primary_v1", "mse_s43", "mse_s44"],
    "Utilitarian, 250": ["utilitarian_mse_s42_v2", "utilitarian_mse_s43", "utilitarian_mse_s44"],
}
SEED = {"mse_short_primary_v1": 42, "mse_s43": 43, "mse_s44": 44,
        "utilitarian_mse_s42_v2": 42, "utilitarian_mse_s43": 43, "utilitarian_mse_s44": 44}


def dev(arm):
    p = R / "arms" / arm / "trainer_state.json"
    if not p.exists():
        return None
    e = [h for h in json.loads(p.read_text())["log_history"] if "eval_nbpo/nmse" in h]
    return e[-1] if e else None


def test_surplus(arm):
    for d in sorted(glob.glob(str(R / "evaluations/saferlhf_*/"))):
        p = os.path.join(d, arm, "per_prompt.jsonl")
        if os.path.isfile(p):
            return [json.loads(l) for l in open(p) if json.loads(l)["split"] == "test"]


for family, arms in FAMILIES.items():
    print(f"=== {family} ===")
    worst, nmse, sign = [], [], []
    for arm in arms:
        e, rows = dev(arm), test_surplus(arm)
        if e is None:
            print(f"  s{SEED[arm]}  not trained"); continue
        line = (f"  s{SEED[arm]}  nmse={e['eval_nbpo/nmse']:.4f}  sign={e['eval_nbpo/sign_accuracy']:.4f}  "
                f"pearson={e['eval_nbpo/pearson']:+.4f}")
        nmse.append(e["eval_nbpo/nmse"]); sign.append(e["eval_nbpo/sign_accuracy"])
        if rows:
            s = np.array([r["s_original_reference"] for r in rows], float)
            worst.append(s.mean(0).min())
            line += (f"  | s=({s.mean(0)[0]:+.4f},{s.mean(0)[1]:+.4f})  worst={s.mean(0).min():+.4f}"
                     f"  both>0={(s > 0).all(1).mean():.3f}")
        else:
            line += "  | test evaluation pending"
        print(line)
    if len(nmse) > 1:
        out = f"  over {len(nmse)} seeds: nmse={statistics.fmean(nmse):.4f}±{statistics.stdev(nmse):.4f}"
        out += f"  sign={statistics.fmean(sign):.4f}±{statistics.stdev(sign):.4f}"
        if len(worst) > 1:
            out += f"  worst={statistics.fmean(worst):+.4f}±{statistics.stdev(worst):.4f}"
        print(out)
    print()

a = [test_surplus(x) for x in FAMILIES["NBPO-MSE, 250"]]
b = [test_surplus(x) for x in FAMILIES["Utilitarian, 250"]]
if all(a) and all(b):
    print("=== Nash minus utilitarian, paired per prompt, seed-matched ===")
    rng = np.random.default_rng(20260909)
    for i, seed in enumerate((42, 43, 44)):
        ib = {r["prompt_id"]: r for r in b[i]}
        pr = [(x, ib[x["prompt_id"]]) for x in a[i] if x["prompt_id"] in ib]
        d = np.array([np.array(x["s_original_reference"], float) - np.array(y["s_original_reference"], float)
                      for x, y in pr])
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean(0) for _ in range(2000)])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        print(f"  s{seed}  help={d.mean(0)[0]:+.4f} [{lo[0]:+.4f},{hi[0]:+.4f}]   "
              f"harm={d.mean(0)[1]:+.4f} [{lo[1]:+.4f},{hi[1]:+.4f}]")
    print("  (prompt-sampling intervals; a CI containing zero is not an equivalence test)")
