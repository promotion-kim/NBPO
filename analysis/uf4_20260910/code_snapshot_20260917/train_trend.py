"""Trend of the NBPO projection diagnostics within the CURRENT run only."""
import re, sys, json
path = sys.argv[1]
marker = "launching uf4_train_nbpo_mse_s42"
text = open(path, errors="ignore").read()
segments = text.split(marker)
cur = segments[-1]                      # everything after the newest launch
rows = []
for m in re.finditer(r"\{'loss': ([0-9.eE+-]+).*?'grad_norm': ([0-9.eE+-]+).*?"
                     r"'rewards/accuracies': ([0-9.eE+-]+).*?"
                     r"'nbpo/h_abs': ([0-9.eE+-]+).*?'nbpo/target_abs': ([0-9.eE+-]+).*?"
                     r"'nbpo/residual_abs': ([0-9.eE+-]+).*?'epoch': ([0-9.eE+-]+)", cur, re.S):
    loss, gn, acc, h, t, r, ep = (float(x) for x in m.groups())
    rows.append({"epoch": ep, "loss": loss, "grad_norm": gn, "acc": acc,
                 "h_abs": h, "target_abs": t, "residual_abs": r})
print("logged steps in current run:", len(rows))
if not rows:
    sys.exit(0)
n = len(rows)
for label, sl in (("first 20", rows[:20]), ("middle 20", rows[n//2-10:n//2+10]), ("last 20", rows[-20:])):
    k = len(sl)
    avg = lambda key: sum(x[key] for x in sl)/k
    # ratio of residual magnitude to target magnitude: 1.0 means no fit at all
    print("%-10s loss %8.2f  |h| %5.2f  |target| %5.2f  |resid| %5.2f  "
          "resid/target %.3f  acc %.3f  gradnorm %9.0f"
          % (label, avg("loss"), avg("h_abs"), avg("target_abs"), avg("residual_abs"),
             avg("residual_abs")/max(avg("target_abs"), 1e-9), avg("acc"), avg("grad_norm")))
evals = re.findall(r"'eval_loss': ([0-9.eE+-]+)", cur)
print("eval_loss in current run:", evals)
