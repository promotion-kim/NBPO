#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {"instruction_following":"Instruction\nfollowing","truthfulness":"Truthfulness","honesty":"Honesty","helpfulness":"Helpfulness","safety":"Safety"}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--summary",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); a=ap.parse_args()
    d=json.loads(a.summary.read_text()); objs=d["protocol"]["objectives"]; ang=np.linspace(0,2*np.pi,len(objs),endpoint=False).tolist(); ang+=ang[:1]
    fig,ax=plt.subplots(figsize=(6.0,4.8),subplot_kw={"polar":True})
    baseline_color="#999999"
    for name,r in d["methods"].items():
        vals=[r["mean_by_objective"][o] for o in objs]; vals+=vals[:1]
        if name=="Base": color="#111111"; lw=2.0; alpha=1; z=5
        elif name=="RMOD": color="#e07b16"; lw=2.1; alpha=1; z=5
        elif name.startswith("RONPO"):
            stage=int(name.split("S")[-1]); color=plt.cm.Blues(.38+.13*stage); lw=1.3+.25*stage; alpha=.95; z=6
        else: color=baseline_color; lw=.9; alpha=.55; z=2
        ax.plot(ang,vals,lw=lw,alpha=alpha,color=color,label=name,zorder=z)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels([LABELS[o] for o in objs],fontsize=8)
    ax.set_title("UltraFeedback: RONPO stages and RMOD",fontsize=11,pad=15)
    ax.grid(ls=":",alpha=.4); ax.legend(loc="upper left",bbox_to_anchor=(1.02,1.10),fontsize=6.5,frameon=False)
    fig.tight_layout(); a.out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(a.out,bbox_inches="tight"); fig.savefig(a.out.with_suffix(".png"),dpi=200,bbox_inches="tight")

if __name__=="__main__": main()
