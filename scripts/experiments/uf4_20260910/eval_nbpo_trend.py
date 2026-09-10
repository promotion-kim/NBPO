"""Pull the trainer's own held-out NBPO diagnostics from the current run."""
import re, sys
cur = open(sys.argv[1], errors="ignore").read().split("launching uf4_train_nbpo_mse_s42")[-1]
found = sorted(set(re.findall(r"'(eval_nbpo/[a-z_*]+)'", cur)))
print("eval_nbpo keys present:", found or "none")
for k in found + ["eval_loss"]:
    vals = re.findall(re.escape(k) + r"': ([0-9.eE+-]+)", cur)
    if vals:
        print("%-28s %s" % (k, "  ".join("%.4f" % float(v) for v in vals)))
