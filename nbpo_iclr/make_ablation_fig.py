"""Worst-objective trajectory for the three anchoring/pairing variants (HelpSteer2, Qwen3-32B judge)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fixed  = {2: .506, 3: .441, 4: .372, 5: .404}
sibling = {1: .442, 2: .496, 3: .437}
bon = {1: .515, 2: .542, 3: .542, 4: .550, 5: .542, 6: .565, 7: .566}
fig, ax = plt.subplots(figsize=(4.2, 2.9))
for d, lab, st in ((fixed, "anchor fixed at $\\mu$", "o--"), (sibling, "moving anchor, random pair", "s--"),
                   (bon, "moving anchor, best-of-4 (ours)", "D-")):
    ax.plot(sorted(d), [d[k] for k in sorted(d)], st, ms=4, lw=1.5, label=lab)
ax.axhline(.5, color="gray", lw=.8, ls=":")
ax.text(6.6, .505, "reference", color="gray", fontsize=7, va="bottom", ha="right")
ax.set_xlabel("round"); ax.set_ylabel("worst objective  $\\min_k u_k$")
ax.set_ylim(.35, .60); ax.set_xticks(range(1, 8))
ax.legend(fontsize=7, loc="lower right", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("figures/ablation_rounds.pdf")
print("figures/ablation_rounds.pdf written")
