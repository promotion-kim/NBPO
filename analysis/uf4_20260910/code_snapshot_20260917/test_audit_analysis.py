"""Fixtures the audit analysis must get right before it touches real judgments."""
import sys, json
sys.path.insert(0, "/work/uf4_20260910/code")
from analyze_audit import (graph_from, directed_triangles, has_cycle, every_response_loses,
                           edge_estimates, PAIRS, DRAWS_PER_ORDER)

fails = []
def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (" | " + detail if detail else ""))
    if not ok: fails.append(name)

# a strict transitive order 0>1>2>3
trans = {(0,1):0.9,(0,2):0.9,(0,3):0.9,(1,2):0.9,(1,3):0.9,(2,3):0.9}
g = graph_from(trans)
check("transitive: no cycle", not has_cycle(g))
check("transitive: no triangle", len(directed_triangles(g)) == 0)
check("transitive: not every response loses", not every_response_loses(g))

# a 3-cycle 0->1->2->0 with 3 beaten by all
cyc = {(0,1):0.9,(1,2):0.9,(0,2):0.1,(0,3):0.9,(1,3):0.9,(2,3):0.9}
g = graph_from(cyc)
check("3-cycle detected", has_cycle(g))
check("3-cycle: exactly one triangle", len(directed_triangles(g)) == 1, str(directed_triangles(g)))
# 0<2, 1<0, 2<1 and 3 loses to all three: every response does have a loss, so
# there is no strict winner even though response 3 never wins.
check("3-cycle plus a response that loses to all: every response loses",
      every_response_loses(g))

# A 4-cycle with no 3-cycle needs the two diagonals tied: over four nodes every
# COMPLETE tournament containing a cycle also contains a directed triangle, so a
# pure length-four cycle only exists when ties remove edges. This is exactly why
# the protocol counts length-four cycles separately.
four = {(0,1):0.9,(1,2):0.9,(2,3):0.9,(0,3):0.1,(0,2):0.5,(1,3):0.5}
g = graph_from(four)
check("4-cycle: no directed triangle", len(directed_triangles(g)) == 0, str(directed_triangles(g)))
check("4-cycle detected as a cycle", has_cycle(g))
check("4-cycle: every response loses", every_response_loses(g))

# exact ties create no edge
tied = dict(trans); tied[(0,1)] = 0.5
g = graph_from(tied)
check("exact tie creates no edge", (0,1) not in g and (1,0) not in g, str(sorted(g)))

# one unresolved pair makes the whole tournament unresolved
miss = dict(trans); miss[(2,3)] = None
check("unresolved pair -> unresolved tournament", graph_from(miss) is None)

# the order-averaged estimator, and incomplete draws
data = {}
for pair in PAIRS:
    for order in (0,1):
        data[("r","p","A",pair,order)] = [1.0]*DRAWS_PER_ORDER
p = edge_estimates(data, "r", "p", "A")
check("all wins give p_hat 1.0", all(abs(v-1.0) < 1e-12 for v in p.values()))
data[("r","p","A",(0,1),1)] = [0.0]*DRAWS_PER_ORDER
p = edge_estimates(data, "r", "p", "A")
check("order averaging: 1.0 and 0.0 give 0.5", abs(p[(0,1)] - 0.5) < 1e-12, str(p[(0,1)]))
data[("r","p","A",(0,2),0)] = [1.0]*(DRAWS_PER_ORDER-1)
p = edge_estimates(data, "r", "p", "A")
check("nine of ten draws leaves the edge unresolved", p[(0,2)] is None)

print(json.dumps({"failures": fails}))
sys.exit(1 if fails else 0)
