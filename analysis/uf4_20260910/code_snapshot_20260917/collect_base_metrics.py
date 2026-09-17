"""Pull every base-model benchmark number from its own artifact, with provenance."""
import json, glob, os
from pathlib import Path

R = Path("/work/nbpo_repair_20260909")
out = {}

def take(name, path, metric_key, extra=None):
    p = Path(path)
    if not p.exists():
        out[name] = {"status": "artifact not found", "path": str(p)}
        return
    d = json.loads(p.read_text())
    m = d.get("metrics", {})
    if metric_key not in m:
        out[name] = {"status": "metric key absent", "available": sorted(m), "path": str(p)}
        return
    v = m[metric_key]
    out[name] = {"estimate": v["estimate"], "ci95": v.get("ci95"),
                 "n_prompts": v.get("n_prompts"), "denominator": v.get("denominator"),
                 "uncertainty_scope": v.get("uncertainty_scope"),
                 "bootstrap_repetitions": v.get("bootstrap_repetitions"),
                 "metric_key": metric_key, "path": str(p),
                 "model": (d.get("generation_settings") or {}).get("model"),
                 **(extra or {})}

B = R / "evaluations/base_deterministic_v1"
take("ifeval_strict_prompt", B / "ifeval_base.summary.json", "strict_prompt_accuracy")
take("ifeval_strict_instruction", B / "ifeval_base.summary.json", "strict_instruction_accuracy")
take("gsm8k_em", B / "gsm8k_base.summary.json", "exact_match")

for cand in sorted(glob.glob(str(R / "evaluations/harmbench_*/harmbench_base.summary.json"))):
    take("harmbench", cand, "harmful_success_rate")
    if "estimate" in out.get("harmbench", {}):
        break
    d = json.loads(Path(cand).read_text())
    out["harmbench_keys"] = sorted((d.get("metrics") or {}).keys())
    break

for cand in sorted(glob.glob(str(R / "evaluation/xstest_*/base.summary.json"))):
    d = json.loads(Path(cand).read_text())
    out["xstest_keys"] = {"path": cand, "metrics": sorted((d.get("metrics") or {}).keys())}
    for key in ("safe_refusal_rate", "benign_refusal_rate", "safe_full_refusal"):
        if key in (d.get("metrics") or {}):
            take("xstest_safe_refusal", cand, key)
            break
    break

for cand in sorted(glob.glob(str(R / "evaluations/*/alpaca_base.summary.json"))):
    d = json.loads(Path(cand).read_text())
    out["alpaca_keys"] = {"path": cand, "metrics": sorted((d.get("metrics") or {}).keys())}
    break

print(json.dumps(out, indent=1))
