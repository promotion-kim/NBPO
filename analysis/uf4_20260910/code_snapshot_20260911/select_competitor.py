"""Apply the declared competitor-selection rule to whatever dev results exist.

Rule, declared 2026-09-12 02:45 before any of this was measured: the competitor
is the arm with the highest minimum-objective win rate against the base under
the frozen independent judge on policy_dev, among the trained non-bank arms
available when selection runs. Ties beyond 0.002 on the minimum are broken by
average win rate, then by arm name.

It refuses to look at anything else. In particular it never reads
evaluation/final_eval for a non-devsel arm, because those are the final outcomes
the contract forbids this decision from accessing.
"""
import json, pathlib, sys

R = pathlib.Path("/work/uf4_20260910")
CRIT = ("instruction_following", "truthfulness", "honesty", "helpfulness")
rows = []
for d in sorted((R / "evaluation/final_eval").glob("devsel_*")):
    rep_path = d / "complete.json"
    if not rep_path.exists():
        continue
    rep = json.loads(rep_path.read_text())
    wr = {c: rep["criteria"][c]["win_rate_vs_base"] for c in CRIT}
    if any(v is None for v in wr.values()):
        print("skipping incomplete:", d.name); continue
    rows.append({"arm": d.name[len("devsel_"):], "min_objective": min(wr.values()),
                 "average": sum(wr.values()) / len(wr), "per_objective": wr,
                 "n_prompts_complete": rep["criteria"][CRIT[0]]["n_prompts_complete"]})
if not rows:
    print(json.dumps({"status": "no dev-selection results yet"})); sys.exit(0)

rows.sort(key=lambda r: (-r["min_objective"], -r["average"], r["arm"]))
best = rows[0]
close = [r for r in rows[1:] if best["min_objective"] - r["min_objective"] <= 0.002]
decision = {
    "selected_competitor": best["arm"],
    "rule": ("highest minimum-objective win rate against the base, independent judge on "
             "policy_dev; ties within 0.002 broken by average win rate, then arm name"),
    "declared_before_measurement": "analysis/uf4_20260910/crossplay_declarations_20260912.md",
    "candidate_set": [r["arm"] for r in rows],
    "candidates_within_tie_band": [r["arm"] for r in close],
    "ranking": rows,
    "accessed_final_outcomes": False,
    "note": ("candidate set is the trained non-bank arms whose dev judging had finished when "
             "this ran; arms trained later are not retro-fitted into the decision"),
}
out = R / "analysis/crossplay_competitor.json"
out.write_text(json.dumps(decision, indent=2) + "\n")
print(json.dumps({k: v for k, v in decision.items() if k != "ranking"}, indent=2))
for r in rows:
    print("  %-20s min %.4f  avg %.4f  n %d" % (r["arm"], r["min_objective"], r["average"],
                                                r["n_prompts_complete"]))
