import json
LABELS = ("base", "mse_primary_v1", "wbc_primary_v1", "mse_short_primary_v1", "wbc_short_primary_v1")
print(f"{'arm':22s} {'Pearson (clustered CI)':28s} {'sign acc (CI)':26s} {'KL(p*||p_th)':>12s} {'vs center':>10s} {'top1':>6s} {'argmax mass':>12s} {'partial r':>10s}")
for lab in LABELS:
    d = json.load(open(f"/work/nbpo_repair_20260909/analysis_claude/pool_drift_{lab}_dev.json"))
    b = d["pair_statistics_prompt_clustered_bootstrap"]
    def f(k, w):
        v = b[k]
        if v["point"] is None:
            return "-".center(w)
        return f"{v['point']:+.4f} [{v['ci95_low']:+.3f},{v['ci95_high']:+.3f}]".ljust(w)
    pr = d["partial_corr_delta_vs_g_given_length"]
    print(f"{lab:22s} {f('pearson',28)} {f('sign_accuracy',26)} "
          f"{d['kl_target_to_policy_mean']:12.4f} {d['kl_improvement_vs_center']:+10.4f} "
          f"{d['top1_match_rate']:6.3f} {d['mass_on_target_argmax_mean']:12.4f} "
          f"{'-' if pr is None else format(pr, '+.4f'):>10s}")
print()
print("center (reference) values: KL(p*||p_t) = 0.7879, argmax mass = 0.1250, chance top1 = 0.125")
