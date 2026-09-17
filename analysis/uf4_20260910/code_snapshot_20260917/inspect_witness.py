"""Show, for real cyclic triangles in panel A, why each fails the both-orders rule."""
import sys, json
import numpy as np
sys.path.insert(0, "/work/uf4_20260910/code")
from analyze_audit import load, directed_triangles, graph_from, edge_estimates, DRAWS_PER_ORDER

data = load("/work/uf4_20260910/audit/v1/judgments/audit_100/judgments.jsonl")
prompts = sorted({k[1] for k in data})
shown = 0
for rubric in ("instruction_following", "honesty"):
    for prompt in prompts:
        g = graph_from(edge_estimates(data, rubric, prompt, "A"))
        if g is None:
            continue
        tris = directed_triangles(g)
        if not tris:
            continue
        tri = sorted(tris)[0]
        print("== %s  prompt %s  triangle %s" % (rubric, prompt[:10], tri))
        for (u, v) in tri:
            pair = (u, v) if u < v else (v, u)
            m0 = np.mean(data[(rubric, prompt, "A", pair, 0)])
            m1 = np.mean(data[(rubric, prompt, "A", pair, 1)])
            d0 = m0 if pair == (u, v) else 1 - m0
            d1 = m1 if pair == (u, v) else 1 - m1
            print("   edge %d->%d  order0 %.2f  order1 %.2f  avg %.2f   both>0.5? %s"
                  % (u, v, d0, d1, 0.5*(d0+d1), (d0 > 0.5 and d1 > 0.5)))
        shown += 1
        if shown >= 4:
            break
    if shown >= 4:
        break
# how often does ANY single edge agree in both orders?
agree = total = 0
for rubric in ("instruction_following", "truthfulness", "honesty", "helpfulness"):
    for prompt in prompts:
        g = graph_from(edge_estimates(data, rubric, prompt, "A"))
        if g is None: continue
        for tri in directed_triangles(g):
            for (u, v) in tri:
                pair = (u, v) if u < v else (v, u)
                m0 = np.mean(data[(rubric, prompt, "A", pair, 0)])
                m1 = np.mean(data[(rubric, prompt, "A", pair, 1)])
                d0 = m0 if pair == (u, v) else 1 - m0
                d1 = m1 if pair == (u, v) else 1 - m1
                total += 1
                agree += int(d0 > 0.5 and d1 > 0.5)
print(json.dumps({"cycle_edges_examined": total, "edges_agreeing_in_both_orders": agree,
                  "fraction": round(agree/max(total,1), 4)}))
