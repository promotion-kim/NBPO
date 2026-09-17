import json
OBJ = ("instruction_following", "truthfulness", "honesty", "helpfulness")
for name in ("gpm_s42", "bt_s42"):
    r = json.load(open("/work/uf4_20260910/teacher/%s/report.json" % name))
    s = r["selected"]
    print("== %s | best step %d | mean NLL %.4f | worst NLL %.4f | %.0f s | steps %d"
          % (name, s["step"], s["_mean_nll"], s["_worst_nll"], r["seconds"], r["total_steps"]))
    for o in OBJ:
        m = s[o]
        acc = "%.4f" % m["accuracy_excl_ties"] if m["accuracy_excl_ties"] is not None else "n/a"
        print("   %-22s nll %.4f  brier %.4f  acc(excl ties) %s  tie %.3f  n %d"
              % (o, m["nll"], m["brier"], acc, m["tie_fraction"], m["n_labeled"]))
    print("   length budget (train):", json.dumps(r["length_budget"]["train_collator"]))
    print("   length budget (dev):  ", json.dumps(r["length_budget"]["dev_collator"]))
