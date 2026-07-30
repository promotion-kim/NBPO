#!/usr/bin/env python3
"""Merge P7 score shards and evaluate the pre-registered Stage-3 goal."""
from __future__ import annotations
import argparse, csv, json, subprocess, sys
from pathlib import Path
import numpy as np

ARMS = ["base", "ronpo_os_stage3", "ronpo_topmass_stage3", "inpo_avg_stage3", "sppo_avg_stage3", "simpo_stage3", "ipo_stage3", "dpo_stage3", "ht_mnpo_harmless_stage3", "ht_mnpo_helpfulness_stage3"]

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--project",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--lock",type=Path,required=True); a=p.parse_args(); out=a.output
    audit=json.loads((out/"pool_audit.json").read_text()); count=len(audit["eligible_models"]); merge=a.project/"analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py"
    for obj in ["helpfulness","harmlessness"]:
        inputs=[out/"score_shards"/f"{obj}_{i}.jsonl" for i in range(6)]
        subprocess.run([sys.executable,str(merge),"merge","--inputs",*map(str,inputs),"--output",str(out/"scores"/f"{obj}.jsonl"),"--audit",str(out/"scores"/f"{obj}_audit.json"),"--expected-records","1000","--expected-scores-per-row",str(count),"--strip-responses"],check=True)
    aggregate=a.project/"analysis/p5_8b_robust_stage1_stage2_20260717/aggregate_stage1_comparison.py"
    subprocess.run([sys.executable,str(aggregate),"--helpfulness",str(out/"scores/helpfulness.jsonl"),"--harmlessness",str(out/"scores/harmlessness.jsonl"),"--pool-audit",str(out/"pool_audit.json"),"--output-dir",str(out/"results"),"--bootstrap","2000","--seed","42","--scope","fresh 1,000-prompt PKU-SafeRLHF default-test panel, prompt-disjoint from P6b and every listed earlier panel","--report-title","P7 Stage-3 fresh SafeRLHF test comparison","--ronpo-arm","ronpo_os_stage3","--ronpo-arm","ronpo_topmass_stage3","--comparison-label","best eligible non-RONPO trained Stage-3 arm"],check=True)
    s=json.loads((out/"results/ranked_validation_summary.json").read_text()); pp={}
    with (out/"results/per_prompt_metrics.csv").open() as h:
        for r in csv.DictReader(h): pp.setdefault(r["model"],{})[r["prompt_id"]]=r
    ids=sorted(pp["base"]); rng=np.random.default_rng(42); ix=rng.integers(0,len(ids),size=(2000,len(ids))); rows={r["model"]:r for r in s["ranking"]}; goals={}
    for arm in ["ronpo_os_stage3","ronpo_topmass_stage3"]:
        comp=s["new_arm_paired_comparisons"].get(arm)
        if not comp: goals[arm]={"eligible":False,"goal_met":False,"reason":"failed stability gate or absent from eligible scoring pool"}; continue
        avg_other=max((r for k,r in rows.items() if k!="base" and not k.startswith("ronpo_")),key=lambda r:(r["mean_objective_norm_score"],r["model"]))["model"]
        d=np.asarray([float(pp[arm][x]["prompt_avg_norm"])-float(pp[avg_other][x]["prompt_avg_norm"]) for x in ids]); aci=[float(np.quantile(d[ix].mean(1),.025)),float(np.quantile(d[ix].mean(1),.975))]; wci=comp["paired_prompt_worst_difference_ci95"]
        goals[arm]={"eligible":True,"worst_comparator":comp["worst_comparator"],"worst_difference":comp["paired_prompt_worst_difference"],"worst_difference_ci95":wci,"worst_pass":wci[0]>0,"avg_comparator":avg_other,"avg_difference":float(d.mean()),"avg_difference_ci95":aci,"avg_floor_pass":aci[0]>-.02,"goal_met":wci[0]>0 and aci[0]>-.02}
    payload={"status":"complete","goal_rule":json.loads(a.lock.read_text())["goal_rule"],"eligible_models":s["eligible_models"],"failed_models":audit["failed_models"],"ronpo_goal_assessments":goals,"spent_sealed_split_touched":False}; (out/"goal_assessment.json").write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2))

if __name__=="__main__": main()
