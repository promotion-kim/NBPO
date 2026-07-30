#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(); p.add_argument("--summary",type=Path,required=True); p.add_argument("--out",type=Path,required=True); a=p.parse_args()
    d=json.loads(a.summary.read_text()); methods=d["methods"]; primary=d["comparisons"]["primary_combined_s6_minus_rmod_k16"]
    lines=["# UF5 combined continuation: fresh confirmation","","All policies and RMOD were evaluated once on the prompt-disjoint 647-prompt part-3 panel and scored together in one BF16 ArmoRM batch context. No stage was selected by reward.","","| Method | Avg. | Worst | Worst head | Delta vs Base | 95% CI |","|---|---:|---:|---|---:|---:|"]
    for label,r in methods.items():
        lo,hi=r["worst_head_paired_delta_ci95"]
        lines.append(f"| {label} | {r['average_of_objective_means']:.6f} | {r['worst_objective_mean']:.6f} | {r['worst_objective']} | {r['worst_head_paired_delta_vs_base']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lo,hi=primary["ci95"]
    verdict="PASS" if primary["pass"] else "FAIL"
    lines += ["","## Preregistered primary comparison","",f"Combined S6 minus RMOD K=16 on the minimum marginal head mean: {primary['estimate']:+.6f}, paired bootstrap 95% CI [{lo:+.6f}, {hi:+.6f}]. **{verdict}**.","", "The result is reported as measured. A null or negative comparison is not retried or replaced by a secondary endpoint.",""]
    a.out.write_text("\n".join(lines))


if __name__ == "__main__": main()
