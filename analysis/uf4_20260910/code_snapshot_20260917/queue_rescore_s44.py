"""Queue the four cross-arm capability panels with PROSPER's third seed.

The arm does not exist yet -- its training has 770 steps left -- so each panel
declares depends_on the generation job that writes that family's responses for
it. The controller then dispatches the scorers the moment those generations
land, instead of waiting for an operator to notice. The scorers are one GPU and
a few minutes each, and they sit at priority 60, ahead of the DPO weight trains
at 73-76, so a finished main-body row is not held behind ten hours of sweep.

Each panel is a new job id carrying its new arm count, and its --out path is
renamed the same way, so the artifacts the current tables were built from stay
on disk and the host filler (which takes the highest arm count) picks up the
new one only once it is complete.
"""
import json
import re
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
QUEUE = ROOT / "jobs/queue"
NEW_ARM = "prosper_mse_s44"
# panel source spec, label prefix, the generation job that feeds it
PANELS = [("uf4_score_harmbench_22arm", "uf4hb", "uf4_genhb3_prosper_mse_s44"),
          ("uf4_score_xstest_21arm", "uf4xs", "uf4_genxs_prosper_mse_s44"),
          ("uf4_score_alpaca_arena_21arm", "uf4ab", "uf4_genab_prosper_mse_s44"),
          ("uf4_score_mtbench_22arm", "uf4mt", "uf4_genmt_prosper_mse_s44")]


def specs():
    out = {}
    for path in QUEUE.glob("*.json"):
        spec = json.loads(path.read_text())
        out[spec["job_id"]] = spec
    return out


def main():
    live = specs()
    for source, prefix, feeder in PANELS:
        old = live.get(source)
        if old is None:
            print("missing source spec: %s" % source)
            continue
        if feeder not in live:
            print("no generation job %s; not queueing %s" % (feeder, source))
            continue
        cmd = list(old["command"])
        i = cmd.index("--labels")
        j = i + 1
        while j < len(cmd) and not cmd[j].startswith("--"):
            j += 1
        labels = cmd[i + 1:j]
        label = "%s_%s" % (prefix, NEW_ARM)
        if label in labels:
            print("already present in %s: %s" % (source, label))
            continue
        labels = sorted(labels + [label])
        # "base" sorts before the prefixed labels, so the count is the list
        # length either way; the id must state the count the panel really has.
        count = len(labels)
        new_id = re.sub(r"\d+arm$", "%darm" % count, source)
        if new_id == source or new_id in live:
            new_id = "%s_%darm" % (re.sub(r"_\d+arm$", "", source), count)
        new_cmd = cmd[:i + 1] + labels + cmd[j:]
        for k, token in enumerate(new_cmd):
            if k > 0 and isinstance(token, str) and re.search(r"\d+arm", token):
                new_cmd[k] = re.sub(r"\d+arm", "%darm" % count, token)
        spec = dict(old)
        spec["job_id"] = new_id
        spec["priority"] = 60
        spec["depends_on"] = [feeder]
        spec["command"] = new_cmd
        spec["artifacts"] = [re.sub(r"\d+arm", "%darm" % count, a)
                             for a in old.get("artifacts", [])]
        if new_id in live:
            print("exists: %s" % new_id)
            continue
        path = QUEUE / ("%d_%s.json" % (spec["priority"], new_id))
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(spec, indent=2) + "\n")
        tmp.replace(path)
        live[new_id] = spec
        print("queued %-34s labels %d -> %d, depends_on %s, out %s"
              % (new_id, count - 1, count, feeder,
                 spec["artifacts"][0] if spec["artifacts"] else "-"))


if __name__ == "__main__":
    main()
