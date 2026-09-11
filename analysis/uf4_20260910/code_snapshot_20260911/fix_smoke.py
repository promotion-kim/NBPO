"""Re-arm the DPO smoke on four cards, where ZeRO-2 can actually shard.

The 1-GPU smoke was my error, not a DPO problem. ZeRO-2 partitions optimizer
state across ranks, so at one rank a single card has to hold the full fp32
master weights, momentum and variance for an 8B model -- about 96 GB -- while
also holding the frozen online reference model. It OOMed asking for 29.92 GiB
inside optimizer.step().

It had already served most of its purpose before dying there: the config loaded,
the dataset materialized and collated, the online reference initialized, and the
DPO loss and backward completed, which is what proves this morning's `counts`
fix. Only the sharded optimizer step went untested, and that step is identical
to the one every working arm uses.

Re-arming at four GPUs costs about five minutes now that the dataset Map is
cached, which is cheap insurance before committing three hours to the real arm.
The DPO trainings shift down one priority so the smoke stays ahead of them.
"""
import json, pathlib

Q = pathlib.Path("/work/uf4_20260910/jobs/queue")
SHIFT = {"uf4_train_dpo_uniform_mse_s42": 68, "uf4_train_dpo_if_only_mse_s42": 69,
         "uf4_train_dpo_truth_only_mse_s42": 70, "uf4_train_dpo_honesty_only_mse_s42": 71,
         "uf4_train_dpo_help_only_mse_s42": 72, "uf4_train_dpo_help_heavy_mse_s42": 73,
         "uf4_train_dpo_truth_heavy_mse_s42": 74}
S = json.load(open("/work/uf4_20260910/jobs/state.json"))["jobs"]

for path in sorted(Q.glob("*.json")):
    spec = json.loads(path.read_text())
    jid = spec["job_id"]
    if jid == "uf4_train_smoke_dpo_uniform":
        spec["gpus"] = 4
        spec["command"] = [("--nproc_per_node=4" if c.startswith("--nproc_per_node") else c)
                           for c in spec["command"]]
        spec["priority"] = 67
        spec["retry"] = int(spec.get("retry", 0)) + 1
        spec["retry_reason"] = ("1-GPU ZeRO-2 cannot shard the 8B fp32 optimizer state and "
                                "OOMed in optimizer.step(); rerun on four cards, which is the "
                                "configuration every real arm uses")
        path.write_text(json.dumps(spec, indent=2) + "\n")
        print("re-armed %s: gpus 1 -> 4, priority -> 67, retry %d" % (jid, spec["retry"]))
    elif jid in SHIFT and S.get(jid, {}).get("state") not in ("RUNNING", "DONE"):
        if spec.get("priority") != SHIFT[jid]:
            old = spec.get("priority")
            spec["priority"] = SHIFT[jid]
            path.write_text(json.dumps(spec, indent=2) + "\n")
            print("  %-40s %s -> %d" % (jid, old, SHIFT[jid]))
