"""Rescore the four cross-arm capability panels with the MOPO and PROSPER arms.

Their benchmark generations are already on disk -- 12 directories for MOPO's
three seeds and 8 for PROSPER's two -- so the only missing step is the scoring
pass, which is what leaves Table 3's safety and arena cells pending for those
two rows. Each scorer is a single-GPU job of a few minutes, and three cards are
free while the last training arm is judged.

The new label list is the old one plus the arms that now exist, so the panel
grows rather than being replaced, and the previous artifacts are left in place
under their own arm-count names.
"""
import json
import re
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
QUEUE = ROOT / "jobs/queue"
NEW_ARMS = ["mopo_mse_s42", "mopo_mse_s43", "mopo_mse_s44",
            "prosper_mse_s42", "prosper_mse_s43"]
SOURCES = [("uf4_score_harmbench_16arm", "uf4hb", "harmbench"),
           ("uf4_score_xstest_16arm", "uf4xs", "xstest"),
           ("uf4_score_alpaca_arena_16arm", "uf4ab", "alpaca_arena"),
           ("uf4_score_mtbench_17arm", "uf4mt", "mtbench")]


def find(job_id):
    for path in QUEUE.glob("*.json"):
        spec = json.loads(path.read_text())
        if spec["job_id"] == job_id:
            return spec
    return None


def write(spec):
    existing = {json.loads(p.read_text())["job_id"] for p in QUEUE.glob("*.json")}
    if spec["job_id"] in existing:
        return "exists  " + spec["job_id"]
    path = QUEUE / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(spec, indent=2) + "\n")
    tmp.replace(path)
    return "queued  " + spec["job_id"]


def main():
    for job_id, prefix, family in SOURCES:
        old = find(job_id)
        if old is None:
            print("missing source spec: %s" % job_id)
            continue
        cmd = list(old["command"])
        if "--labels" not in cmd:
            print("no --labels in %s" % job_id)
            continue
        i = cmd.index("--labels")
        # labels run until the next flag
        j = i + 1
        while j < len(cmd) and not cmd[j].startswith("--"):
            j += 1
        labels = cmd[i + 1:j]
        added = ["%s_%s" % (prefix, a) for a in NEW_ARMS
                 if "%s_%s" % (prefix, a) not in labels]
        if not added:
            print("nothing to add for %s" % job_id)
            continue
        labels = sorted(labels + added)
        count = len(labels)
        new_id = re.sub(r"\d+arm$", "%darm" % count, job_id)
        if new_id == job_id:
            new_id = "%s_%darm" % (job_id, count)
        new_cmd = cmd[:i + 1] + labels + cmd[j:]
        # the output name carries the arm count, so a new panel does not
        # overwrite the one the current tables were built from
        for k, token in enumerate(new_cmd):
            if isinstance(token, str) and re.search(r"\d+arm", token) and k != 0:
                new_cmd[k] = re.sub(r"\d+arm", "%darm" % count, token)
        spec = dict(old)
        spec["job_id"] = new_id
        spec["priority"] = 60
        spec["command"] = new_cmd
        spec["depends_on"] = []
        spec["artifacts"] = [re.sub(r"\d+arm", "%darm" % count, a)
                             for a in old.get("artifacts", [])]
        print("%s  labels %d -> %d, added %s" % (write(spec), len(labels) - len(added),
                                                 count, ",".join(added)))


if __name__ == "__main__":
    main()
