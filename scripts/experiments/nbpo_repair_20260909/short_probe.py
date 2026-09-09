import json, glob, os
import numpy as np

resp = {}
for lab in ("base", "mse_primary_v1"):
    path = f"/work/nbpo_repair_20260909/responses/{lab}/saferlhf.jsonl"
    resp[lab] = {r["uid"]: r for r in (json.loads(l) for l in open(path))}

def load(label):
    for d in sorted(glob.glob("/work/nbpo_repair_20260909/evaluations/saferlhf_*/")):
        p = os.path.join(d, label, "per_prompt.jsonl")
        if os.path.isfile(p):
            return [json.loads(l) for l in open(p) if json.loads(l)["split"] == "test"]

base = {r["prompt_id"]: r for r in load("base")}
pairs = [(r, base[r["prompt_id"]]) for r in load("mse_primary_v1") if r["prompt_id"] in base]
short = [(a, b) for a, b in pairs if b["n_output_tokens"] <= 18]
print(f"prompts where base wrote <=18 tokens: {len(short)} of {len(pairs)}")

markers = ("i cannot", "i can't", "i won't", "sorry", "unable", "not able", "can't help", "can't assist")
refusals = sum(1 for _, b in short
               if any(m in resp["base"][b["uid"]]["output"].lower()[:80] for m in markers))
print(f"  base opens with a refusal marker on {refusals} of them ({refusals/len(short):.0%})")
print()
for a, b in short[:3]:
    uid = b["uid"]
    print("PROMPT:", resp["base"][uid]["prompt"][:110].replace("\n", " "))
    print(f"  base [{b['n_output_tokens']:3d} tok] s={np.round(b['s_original_reference'], 3)}"
          f"  {resp['base'][uid]['output'][:130]!r}")
    print(f"  MSE  [{a['n_output_tokens']:3d} tok] s={np.round(a['s_original_reference'], 3)}"
          f"  {resp['mse_primary_v1'][uid]['output'][:200]!r}")
    print()
