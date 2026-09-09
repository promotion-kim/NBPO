"""How much of the canonical target is explained by response length alone."""
import json, sys, numpy as np
from pathlib import Path

def load(path):
    g, L, P, R, pid = [], [], [], [], []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        g.append(row["g"]); L.append(row["response_tokens"])
        P.append(row["p_star"]); R.append(row["refusal_keyword_diagnostic"])
        pid.append(row["prompt_id"])
    return (np.array(g), np.array(L, dtype=float), np.array(P), np.array(R, dtype=float), pid)

def report(name, path):
    g, L, P, R, pid = load(path)
    n, K = g.shape
    out = {"split": name, "n_prompts": int(n), "n_candidates": int(K)}
    # within-prompt centering: the pair target is g_a - g_b, so only within-prompt
    # variation matters.
    gc = g - g.mean(1, keepdims=True)
    Lc = L - L.mean(1, keepdims=True)
    Rc = R - R.mean(1, keepdims=True)
    def corr(a, b):
        a, b = a.ravel(), b.ravel()
        if a.std() == 0 or b.std() == 0: return None
        return float(np.corrcoef(a, b)[0, 1])
    out["corr_within_prompt_g_vs_length"] = corr(gc, Lc)
    out["corr_within_prompt_g_vs_refusal"] = corr(gc, Rc)
    out["corr_within_prompt_pstar_vs_length"] = corr(P - P.mean(1, keepdims=True), Lc)
    # r^2 of the best length-only linear predictor of the within-prompt target
    c = out["corr_within_prompt_g_vs_length"]
    out["length_only_r2_of_target"] = None if c is None else c * c
    out["entropy_mean"] = float(np.mean(-(P * np.log(np.clip(P, 1e-12, None))).sum(1)))
    out["ess_mean"] = float(np.mean(1.0 / (P ** 2).sum(1)))
    out["max_mass_mean"] = float(P.max(1).mean())
    out["frac_prompts_max_mass_over_0.5"] = float((P.max(1) > 0.5).mean())
    out["length_mean"] = float(L.mean()); out["length_std_within_prompt"] = float(Lc.std())
    out["g_std_within_prompt"] = float(gc.std())
    return out

root = Path("/work/nbpo_repair_20260909/teachers/nash_repair_v2")
results = [report(s, root / s / "teacher_per_prompt.jsonl") for s in ("train", "dev", "test")]
Path("/work/nbpo_repair_20260909/analysis_claude/teacher_length_structure.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
