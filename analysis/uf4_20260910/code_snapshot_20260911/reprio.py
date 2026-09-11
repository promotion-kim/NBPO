"""Put the main-body critical path ahead of the P3 lm-eval backfill.

The controller orders by the spec's `priority` field, not by the file name, so
editing the field in place reprioritises a job without creating a second spec
with the same job_id -- which would make load_specs() raise and stall the queue.
Only PENDING/READY jobs are touched; a RUNNING or DONE job is left alone.
"""
import json, pathlib
Q = pathlib.Path("/work/uf4_20260910/jobs/queue")
STATE = json.load(open("/work/uf4_20260910/jobs/state.json"))["jobs"]

NEW = {
    "uf4_finaleval_gen_util_mse_s44_shard2": 73,
    "uf4_finaleval_gen_util_mse_s44_shard3": 73,
    "uf4_finaleval_judge_util_mse_s44": 74,
    "uf4_finaleval_gen_fixedref_mse_s42_shard0": 75,
    "uf4_finaleval_gen_fixedref_mse_s42_shard1": 75,
    "uf4_finaleval_gen_fixedref_mse_s42_shard2": 75,
    "uf4_finaleval_gen_fixedref_mse_s42_shard3": 75,
    "uf4_finaleval_judge_fixedref_mse_s42": 76,
}
for path in sorted(Q.glob("*.json")):
    spec = json.loads(path.read_text())
    jid = spec["job_id"]
    if jid not in NEW:
        continue
    state = STATE.get(jid, {}).get("state")
    if state in ("RUNNING", "DONE"):
        print("skip (%s): %s" % (state, jid)); continue
    old = spec.get("priority")
    if old == NEW[jid]:
        print("already: %s" % jid); continue
    spec["priority"] = NEW[jid]
    path.write_text(json.dumps(spec, indent=2) + "\n")
    print("%-46s %s -> %s  (state=%s)" % (jid, old, NEW[jid], state))
