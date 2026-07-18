import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Normalized worst-reward per method per stage, from the existing local per-stage evals.
# Stage1: p5 stage1/fixed_p4_validation (49 prompts). Stage2: p5 stage2 fixed_p4 diagnostic
# (49 prompts; INCLUDES the stability-gate-failed RONPO arm). Stage3: p7 fresh default-test (1000 prompts).
data = {
 "RONPO":            [0.304, 0.404, 0.433],
 "IPO":              [0.431, 0.414, 0.427],
 "INPO-avg":         [0.405, 0.379, 0.389],
 "SimPO":            [0.363, 0.394, 0.373],
 "DPO":              [0.352, 0.321, 0.358],
 "HT-MNPO (help.)":  [0.242, 0.303, 0.287],
 "HT-MNPO (harml.)": [0.274, 0.241, 0.285],
 "SPPO-avg":         [0.280, 0.263, 0.285],
 "Base":             [0.097, 0.089, 0.128],
}
stages=[1,2,3]
fig,ax=plt.subplots(figsize=(7,5))
for name,ys in data.items():
    if name=="RONPO":
        ax.plot(stages,ys,marker="o",lw=3.0,ms=9,color="#d62728",zorder=5,label=name)
    elif name=="IPO":
        ax.plot(stages,ys,marker="s",lw=2.0,ms=7,color="#1f77b4",zorder=4,label=name)
    elif name=="Base":
        ax.plot(stages,ys,marker="x",lw=1.3,ms=6,color="0.5",ls="--",label=name)
    else:
        ax.plot(stages,ys,marker=".",lw=1.3,ms=6,alpha=0.8,label=name)
ax.set_xticks(stages); ax.set_xlabel("Training stage"); ax.set_ylabel("Worst-objective reward (per-prompt normalized)")
ax.set_title("SafeRLHF: worst-objective reward across training stages")
ax.grid(alpha=0.25); ax.legend(fontsize=8,ncol=2,loc="lower right")
fig.tight_layout()
out="/tmp/claude-1330802213/-home-sjkim-MNPO/d40b2775-22b8-40f7-822d-b64571925031/scratchpad/worst_vs_stage.png"
fig.savefig(out,dpi=170); print("saved",out)
# rank of RONPO per stage
for i,s in enumerate(stages):
    order=sorted(data.items(),key=lambda kv:-kv[1][i])
    rank=[n for n,_ in order].index("RONPO")+1
    print(f"stage {s}: RONPO worst={data['RONPO'][i]:.3f} rank={rank}/{len(data)}  (IPO={data['IPO'][i]:.3f})")
