#!/usr/bin/env python3
"""Two LM figures for main_v5: (1) HH helpful-vs-harmless anchored-utility trajectories
showing the bargaining rules iterate into the individual-rationality quadrant while the
baselines iterate out of it; (2) UF radar of the four ArmoRM attributes. Data are the
measured u_k = P_k(pi > mu) win rates against the SFT reference (HH: seed 42 across
iteration rounds; UF: three-seed means)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

C = {"uniform": "#8c8c8c", "os": "#4c72b0", "ks": "#dd8452", "nbpo": "#c44e52"}
LBL = {"uniform": "Utilitarian (avg)", "os": "Egalitarian (maxmin)",
       "ks": "NBPO-KS", "nbpo": "NBPO-NBS"}

# ---------- Figure 3: HH radar, three judges (general LLM oracle, iterated) ----------
HHATTR = ["helpfulness", "harmlessness", "honesty"]
HH = {  # length-neutral HHH panel, general LLM oracle, iterated (u_k vs SFT reference)
    "uniform": [0.772, 0.694, 0.687],
    "nbpo":    [0.726, 0.734, 0.733],
    "ks":      [0.734, 0.732, 0.722],
    "os":      [0.724, 0.734, 0.727],
}
Nh = len(HHATTR)
angh = np.linspace(0, 2 * np.pi, Nh, endpoint=False).tolist(); angh += angh[:1]
fig = plt.figure(figsize=(3.5, 3.1))
ax = fig.add_subplot(111, polar=True)
ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
ax.set_xticks(angh[:-1]); ax.set_xticklabels(HHATTR, fontsize=8)
ax.set_ylim(0.5, 0.80)
ax.set_yticks([0.6, 0.7]); ax.set_yticklabels(["0.6", "0.7"], fontsize=7)
ax.plot(np.linspace(0, 2 * np.pi, 200), [0.5] * 200, color="#3a7d44", lw=1.0, ls="--", zorder=2)
for k in ["uniform", "os", "ks", "nbpo"]:
    v = HH[k] + HH[k][:1]
    lw = 2.6 if k == "nbpo" else 1.2
    z = 5 if k == "nbpo" else 3
    ax.plot(angh, v, color=C[k], lw=lw, label=LBL[k].replace("--", "-"), zorder=z)
    ax.fill(angh, v, color=C[k], alpha=0.14 if k == "nbpo" else 0.04, zorder=z - 1)
ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.13), fontsize=6.5, frameon=False)
fig.tight_layout(); fig.savefig("figures/hh_pareto.pdf", bbox_inches="tight"); plt.close(fig)
print("wrote figures/hh_pareto.pdf")

# ---------- Figure 2: SafeRLHF radar (help/harm/honesty) ----------
ATTR = ["helpfulness", "harmlessness", "honesty"]
UF = {  # SafeRLHF panel, general LLM oracle (u_k vs SFT reference), three-seed means
    "uniform": [0.780, 0.721, 0.734],
    "nbpo":    [0.739, 0.742, 0.750],
    "ks":      [0.743, 0.751, 0.757],
    "os":      [0.669, 0.766, 0.755],
}
N = len(ATTR)
ang = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
ang += ang[:1]
fig = plt.figure(figsize=(3.5, 3.1))
ax = fig.add_subplot(111, polar=True)
ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(ATTR, fontsize=8)
ax.set_ylim(0.6, 0.80)
ax.set_yticks([0.7, 0.8]); ax.set_yticklabels(["0.7", "0.8"], fontsize=7)
for k in ["uniform", "os", "nbpo", "ks"]:
    v = UF[k] + UF[k][:1]
    lw = 2.6 if k == "ks" else 1.2
    z = 5 if k == "ks" else 3
    ax.plot(ang, v, color=C[k], lw=lw, label=LBL[k].replace("--", "-"), zorder=z)
    ax.fill(ang, v, color=C[k], alpha=0.14 if k == "ks" else 0.04, zorder=z - 1)
ax.set_title("", pad=10)
ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.14), fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig("figures/uf_radar.pdf", bbox_inches="tight"); plt.close(fig)
print("wrote figures/uf_radar.pdf")
