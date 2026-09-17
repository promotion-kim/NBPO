"""Emit per-arm policy and generation/evaluation cost for the cost table.

Policy is the trainer-reported runtime times the four devices held, per seed --
the caption's convention -- not the controller's wall total, which also covers
model loading and checkpointing and is quoted separately in the text.
"""
import glob
import json
from pathlib import Path

ROOT = "/work/uf4_20260910"
policy = {}
for p in glob.glob(ROOT + "/arms/*/train_results.json"):
    arm = Path(p).parent.name
    if "." in arm or arm.startswith("smoke"):
        continue
    rt = json.load(open(p)).get("train_runtime")
    if rt:
        policy[arm] = rt * 4 / 3600.0

cost = json.load(open(ROOT + "/analysis/feedback_cost.json"))
gen = {k: v["gpu_h"] for k, v in cost["per_arm_gen_eval"].items()}
print(json.dumps({"policy": policy, "gen_eval": gen}))
