"""Put the fixed-reference seeds first, and give every arm an evaluation chain.

Two separate problems.

First, the pivotal comparison was scheduled fifth. On the seven-arm common set
the single fixed-reference seed leads the three-seed NBPO mean by 3.1 and 3.2
seed standard deviations on truthfulness and honesty, but by only 0.5 on
helpfulness, where it does not even beat NBPO's best seed. A gap of that size
against a standard deviation estimated from three points is exactly what a
second and third seed settle, so those two runs go to the front of the training
order.

Second, maxmin, btrm and all seven DPO arms had no generation or judging queued
behind them. They would each have trained for three hours and finished into an
empty queue, leaving their row unmeasured -- the same gap that was already found
once for fixed-reference seed 42.

Evaluations are numbered below every training. They are PENDING until their own
training finishes, and a PENDING job never reaches the device check, so this
costs a training nothing; it only means that when an arm does finish, its short
generation and judge run before the next three-hour training starts rather than
queueing behind all of them.
"""
import json, pathlib, subprocess, sys

R = "/work/uf4_20260910"
Q = pathlib.Path(R + "/jobs/queue")
STATE = json.load(open(R + "/jobs/state.json"))["jobs"]

# (arm, needs_eval_chain) in the order their evaluations should run
ARMS = [
    ("maxmin_mse_s42", True), ("btrm_mse_s42", True),
    ("fixedref_mse_s43", False), ("fixedref_mse_s44", False),
    ("dpo_uniform_mse_s42", True),
    ("dpo_if_only_mse_s42", True), ("dpo_truth_only_mse_s42", True),
    ("dpo_honesty_only_mse_s42", True), ("dpo_help_only_mse_s42", True),
    ("dpo_help_heavy_mse_s42", True), ("dpo_truth_heavy_mse_s42", True),
]
# trainings, in the order they should occupy the cards
TRAIN_ORDER = [
    ("uf4_train_smoke_dpo_uniform", 63),
    ("uf4_train_fixedref_mse_s43", 64),
    ("uf4_train_fixedref_mse_s44", 65),
    ("uf4_train_btrm_mse_s42", 66),
    ("uf4_train_dpo_uniform_mse_s42", 67),
    ("uf4_train_dpo_if_only_mse_s42", 68),
    ("uf4_train_dpo_truth_only_mse_s42", 69),
    ("uf4_train_dpo_honesty_only_mse_s42", 70),
    ("uf4_train_dpo_help_only_mse_s42", 71),
    ("uf4_train_dpo_help_heavy_mse_s42", 72),
    ("uf4_train_dpo_truth_heavy_mse_s42", 73),
]

# 1. queue the missing evaluation chains
for index, (arm, needs) in enumerate(ARMS):
    if not needs:
        continue
    gen, judge = 40 + 2 * index, 41 + 2 * index
    out = subprocess.run(
        [sys.executable, R + "/code/queue_arm_finaleval.py", "--arm", arm,
         "--depends-on", "uf4_train_" + arm,
         "--gen-priority", str(gen), "--judge-priority", str(judge)],
        capture_output=True, text=True)
    tag = "queued" if out.returncode == 0 else "FAILED"
    print("%-8s eval chain %-26s gen %d judge %d %s"
          % (tag, arm, gen, judge, out.stderr.strip()[:80]))

# 2. renumber the already-queued fixed-reference eval chains into the same scheme
EVAL_PRIORITY = {}
for index, (arm, _) in enumerate(ARMS):
    EVAL_PRIORITY["uf4_finaleval_judge_" + arm] = 41 + 2 * index
    for shard in range(4):
        EVAL_PRIORITY["uf4_finaleval_gen_%s_shard%d" % (arm, shard)] = 40 + 2 * index
TARGET = dict(TRAIN_ORDER, **EVAL_PRIORITY)

changed = 0
for path in sorted(Q.glob("*.json")):
    spec = json.loads(path.read_text())
    jid = spec["job_id"]
    if jid not in TARGET or spec.get("gpus", 0) == 0:
        continue
    state = STATE.get(jid, {}).get("state")
    if state in ("RUNNING", "DONE"):
        print("skip (%s): %s" % (state, jid)); continue
    if spec.get("priority") == TARGET[jid]:
        continue
    old = spec.get("priority")
    spec["priority"] = TARGET[jid]
    path.write_text(json.dumps(spec, indent=2) + "\n")
    print("  %-44s %s -> %d" % (jid, old, TARGET[jid]))
    changed += 1
print("reprioritised %d GPU jobs" % changed)
