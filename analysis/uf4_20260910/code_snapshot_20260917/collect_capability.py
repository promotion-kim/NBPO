"""Read every capability artifact and emit one arm -> {column: value} map.

Runs on the pod. Each column comes from a different artifact family and each
family has been re-run under a new name as arms were added, so the newest
directory wins; the chosen paths are reported so the caller can record them.
"""
import glob
import json
import os
import re
from pathlib import Path

ROOT = "/work/uf4_20260910"
REPAIR = "/work/nbpo_repair_20260909"
out = {}


def put(arm, col, val):
    if val is not None:
        out.setdefault(arm, {})[col] = float(val)


def newest(pattern):
    """Pick the artifact covering the most arms, by parsing the count.

    Each family is re-run under a name carrying its arm count as arms are added,
    so "newest" means "most arms". Sorting by path length then lexically happens
    to work for the counts on disk today -- 12arm loses to 14arm lexically, 9arm
    loses to 14arm on length -- but it is right by accident: 9arm would beat
    14arm if the names were padded, and 100arm would lose to 99arm. Parse the
    number so the choice is correct by construction, and fall back to the path
    for artifacts with no count in the name.
    """
    def key(path):
        m = re.search(r"_(\d+)arm", path)
        return (1, int(m.group(1))) if m else (0, 0)
    hits = [h for h in glob.glob(pattern) if complete(h)]
    return max(hits, key=lambda q: (key(q), q)) if hits else None


def complete(path):
    """Reject an artifact a scorer is still writing.

    The multi-file scorers write one summary per arm as they go and their
    completion marker last, so "newest" is not enough: picking up a directory
    mid-run silently replaced a measured column with dashes. This happened --
    alpaca_arena_uf4_14arm had all fourteen AlpacaEval summaries and one of
    fourteen Arena-Hard ones, and Arena-Hard went to pending for every family.
    A single-file artifact is complete when it exists.
    """
    if os.path.isfile(path):
        return True
    for marker in ("evaluation_complete.json",):
        if os.path.exists(os.path.join(path, marker)):
            return True
    # No marker defined for this family: accept only if nothing newer is being
    # written, which for our layout means the directory has a scores file.
    return any(os.path.exists(os.path.join(path, n)) for n in
               ("mtbench_scores.json", "xstest_local_scores.json"))


for d in glob.glob(REPAIR + "/evaluations/deterministic_uf4_*_v1"):
    arm = Path(d).name.replace("deterministic_uf4_", "").replace("_v1", "")
    for col, bench, key in (("ifeval", "ifeval", "strict_prompt_accuracy"),
                            ("gsm8k", "gsm8k", "exact_match")):
        p = "%s/%s_uf4_%s.summary.json" % (d, bench, arm)
        if os.path.exists(p):
            put(arm, col, json.load(open(p))["metrics"][key]["estimate"])

# The base's own IFEval/GSM8K live beside any arm's run, under the base label.
for d in sorted(glob.glob(REPAIR + "/evaluations/deterministic_uf4_*_v1")):
    for col, bench, key in (("ifeval", "ifeval", "strict_prompt_accuracy"),
                            ("gsm8k", "gsm8k", "exact_match")):
        p = "%s/%s_base.summary.json" % (d, bench)
        if os.path.exists(p) and col not in out.get("base", {}):
            put("base", col, json.load(open(p))["metrics"][key]["estimate"])

hb = newest(ROOT + "/analysis/harmbench_uf4*arm.json") or newest(ROOT + "/analysis/harmbench_uf4.json")
if hb:
    for p in glob.glob(hb + "/*.summary.json"):
        arm = (Path(p).name.replace("harmbench_", "").replace(".summary.json", "")
               .replace("uf4hb_", ""))
        put(arm, "hb", json.load(open(p))["overall"]["estimate"])

xs = newest(ROOT + "/analysis/xstest_local*/xstest_local_scores.json")
if xs:
    for label, v in json.load(open(xs))["labels"].items():
        put(label.replace("uf4xs_", ""), "xs", v["safe"]["2_full_refusal"]["estimate"])

aa = newest(ROOT + "/analysis/alpaca_arena_uf4*")
if aa and os.path.isdir(aa):
    for p in glob.glob(aa + "/alpaca_eval_uf4ab_*.summary.json"):
        arm = Path(p).name.replace("alpaca_eval_uf4ab_", "").replace(".summary.json", "")
        subs = json.load(open(p))["subsets"]
        put(arm, "ae2", subs[sorted(subs)[0]]["estimate"])
    for p in glob.glob(aa + "/arena_hard_uf4ab_*.summary.json"):
        arm = Path(p).name.replace("arena_hard_uf4ab_", "").replace(".summary.json", "")
        subs = json.load(open(p))["subsets"]
        key = "hard500" if "hard500" in subs else sorted(subs)[0]
        put(arm, "ah2", subs[key]["estimate"])

mt = newest(ROOT + "/analysis/mtbench*/mtbench_scores.json")
if mt:
    for label, v in json.load(open(mt))["labels"].items():
        put(label.replace("uf4mt_", ""), "mt", v["score"])

print(json.dumps({"arms": out, "sources": {"harmbench": hb, "xstest": xs,
                                           "alpaca_arena": aa, "mtbench": mt}}))
