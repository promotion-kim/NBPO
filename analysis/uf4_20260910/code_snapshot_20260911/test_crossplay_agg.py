"""Synthetic check of the cross-play aggregator before real judgments exist."""
import itertools, json, pathlib, shutil, subprocess, sys
import numpy as np

ROOT = pathlib.Path("/work/uf4_20260910")
PAIRS = ROOT / "evaluation/crossplay/pairs"
POLICIES = ["base", "pA", "pB", "pC"]
CRIT = ("instruction_following", "truthfulness", "honesty", "helpfulness")
# planted truth: pA slightly better than base, pB clearly better, pC worse
STRENGTH = {"base": 0.0, "pA": 0.3, "pB": 0.8, "pC": -0.4}
rng = np.random.default_rng(7)
made = []
for a, b in itertools.combinations(sorted(POLICIES), 2):
    tag = "%s__vs__%s" % (a, b)
    d = PAIRS / ("SYNTH_" + tag)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    made.append(d)
    rows = []
    for pid in ["p%03d" % i for i in range(120)]:
        for c in CRIT:
            p = 1 / (1 + np.exp(-(STRENGTH[a] - STRENGTH[b])))
            for order in (0, 1):
                v = float(rng.random() < p)
                rows.append({"prompt_id": pid, "criterion": c, "order": order,
                             "a_first": order == 0, "policy_a": a, "policy_b": b,
                             "status": "ok", "value_for_a": v})
    with (d / "judgments.jsonl").open("w") as s:
        for r in rows:
            s.write(json.dumps(r) + "\n")
    (d / "complete.json").write_text(json.dumps(
        {"policy_a": a, "policy_b": b, "judge": "synthetic", "judge_revision": "synthetic"}) + "\n")

out = ROOT / "analysis/crossplay_summary_SYNTH.json"
run = subprocess.run([sys.executable, str(ROOT / "code/aggregate_crossplay.py"),
                      "--bootstrap", "200", "--out", str(out)],
                     capture_output=True, text=True, cwd=str(ROOT))
print(run.stdout[-1500:])
if run.returncode:
    print(run.stderr[-1500:]); sys.exit(1)

rep = json.loads(out.read_text())
M = rep["matrices"]
ok = True
for c in CRIT:
    for i in POLICIES:
        if abs(M[c][i][i] - 0.5) > 1e-12:
            print("FAIL diagonal", c, i, M[c][i][i]); ok = False
        for j in POLICIES:
            if abs(M[c][i][j] + M[c][j][i] - 1.0) > 1e-12:
                print("FAIL antisymmetry", c, i, j); ok = False
st = rep["statistics"]
if abs(st["base"]["W_ref_min"] - 0.5) > 1e-12:
    print("FAIL reference vs itself should be 0.5:", st["base"]["W_ref_min"]); ok = False
if abs(st["base"]["min_s_bank"]) > 1e-12:
    print("FAIL reference bank surplus should be 0:", st["base"]["min_s_bank"]); ok = False
order_ok = st["pB"]["W_ref_min"] > st["pA"]["W_ref_min"] > st["pC"]["W_ref_min"]
if not order_ok:
    print("FAIL planted ordering not recovered"); ok = False
print(json.dumps({"diagonal_and_antisymmetry": "ok" if ok else "FAILED",
                  "reference_self_consistency": "ok",
                  "planted_order_recovered": order_ok,
                  "W_ref_min": {p: round(st[p]["W_ref_min"], 4) for p in POLICIES},
                  "min_s_bank": {p: round(st[p]["min_s_bank"], 4) for p in POLICIES}}, indent=2))
for d in made:
    shutil.rmtree(d)
out.unlink()
print("synthetic fixtures removed")
sys.exit(0 if ok else 1)
