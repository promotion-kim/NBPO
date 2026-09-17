"""Synthetic verdicts with a hand-computed answer, to test analyze_wild.py."""
import hashlib, itertools, json, os
from pathlib import Path
S = Path("/work/sub_20260914")
tag = "wild_smoke_selftest"
out = S / "wild_judgments" / tag
out.mkdir(parents=True, exist_ok=True)
R = list(range(8))
# per (prompt, item): how the eight responses are ordered
MODES = {("p1",0):"order", ("p1",1):"reverse",
         ("p2",0):"order", ("p2",1):"order",
         ("p3",0):"cycle", ("p3",1):"order"}
def value_for_i(mode, i, j):
    """1.0 means response i wins; both orders/draws agree, so p_hat is 0 or 1."""
    if mode == "order":   return 1.0
    if mode == "reverse": return 0.0
    if mode == "cycle":
        # 0>1, 1>2, 2>0 on the top triple; plain order everywhere else
        if (i, j) == (0, 2): return 0.0
        return 1.0
    raise ValueError(mode)
recs = []
for (pid, item), mode in MODES.items():
    for i, j in itertools.combinations(R, 2):
        for order in (0, 1):
            for draw in (0, 1):
                recs.append({"prompt_id": pid, "item": item, "i": i, "j": j,
                             "order": order, "draw": draw, "panel": "A",
                             "verdict": "A", "status": "ok",
                             "value_for_i": value_for_i(mode, i, j)})
chunk = out / "chunk0000.jsonl"
with chunk.open("w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")
h = hashlib.sha256(chunk.read_bytes()).hexdigest()
(out / "chunk0000.manifest.json").write_text(json.dumps(
    {"chunk": 0, "verdicts": len(recs), "sha256": h}) + "\n")
panel = S / "panel" / ("%s.jsonl" % tag)
with panel.open("w") as f:
    for pid in ("p1", "p2", "p3"):
        f.write(json.dumps({"prompt_id": pid, "instruction": "x",
                            "items": [{"text": "a?"}, {"text": "b?"}]}) + "\n")
print("wrote %d synthetic verdicts" % len(recs))

# Verified 2026-09-15 against analyze_wild.py: this construction gives
#   C_single = C_repeated = 1/336 = 0.002976190476190476
#   D_disagreement = 29/84 = 0.34523809523809523  (prompt mean identical here,
#     because the three prompts contribute equal denominators)
#   wildN = 3, item graphs = 6, no-weak-Condorcet = 1/6, order gap 0
# analyze_wild.py reproduced all of these exactly. The synthetic verdicts are
# deleted after the test so they can never be read as measurements; re-run this
# script to recreate them.
