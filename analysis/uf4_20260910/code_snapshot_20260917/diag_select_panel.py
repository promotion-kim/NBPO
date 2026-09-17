"""Freeze the 200 development prompts and the reference occurrences for tonight's
target-to-policy diagnostic, by a rule that cannot see any outcome.

The rule is sha256("uf4-diag-20260914|" + prompt_id): sort ascending and take the
first 200 of the policy-dev split for the evaluation panel. The salt is fixed in
this file, before any score is read, and the chosen IDs are written once.
"""
import glob
import hashlib
import json
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
SALT = "uf4-diag-20260914|"
OUT = ROOT / "analysis/diag_20260914"


def key(text):
    return hashlib.sha256((SALT + text).encode()).hexdigest()


def scan(pool):
    """prompt_id -> {split, roles: {role: {sample_index: record}}} from a cached pool."""
    prompts = {}
    for path in sorted(glob.glob(str(ROOT / "pools" / pool / "shard*/chunk*.jsonl"))):
        with open(path) as stream:
            for line in stream:
                r = json.loads(line)
                slot = prompts.setdefault(r["prompt_id"], {"split": r["split"], "roles": {}})
                slot["roles"].setdefault(r["role"], {})[r["sample_index"]] = {
                    "candidate_id": r["candidate_id"],
                    "response_sha256": r["response_sha256"],
                    "n_tokens": r["n_tokens"],
                }
    return prompts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = scan("dev_v1")
    roles = sorted({role for v in dev.values() for role in v["roles"]})
    counts = {role: sorted({len(v["roles"].get(role, {})) for v in dev.values()})
              for role in roles}
    ordered = sorted(dev, key=key)
    panel = ordered[:200]

    refs, learners = {}, {}
    for pid in panel:
        comparators = dev[pid]["roles"].get("comparator", {})
        picks = sorted(comparators, key=lambda i: key("%s|ref|%d" % (pid, i)))[:4]
        refs[pid] = [comparators[i]["candidate_id"] for i in sorted(picks)]
        learner = dev[pid]["roles"]["learner"]
        learners[pid] = [learner[i]["candidate_id"] for i in sorted(learner)]

    payload = {
        "salt": SALT,
        "rule": "sha256(salt+prompt_id) ascending; first 200 of policy_dev",
        "pool": "dev_v1",
        "roles_seen": roles,
        "occurrences_per_role": counts,
        "n_dev_prompts_available": len(dev),
        "panel_prompt_ids": panel,
        "panel_sha256": hashlib.sha256("|".join(panel).encode()).hexdigest(),
        "reference_occurrences": refs,
        "learner_occurrences": learners,
    }
    (OUT / "panel_dev200.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "dev_prompts_available": len(dev),
        "roles_seen": roles,
        "occurrences_per_role": counts,
        "panel": len(panel),
        "panel_sha256": payload["panel_sha256"][:16],
        "reference_per_prompt": sorted({len(v) for v in refs.values()}),
        "learner_per_prompt": sorted({len(v) for v in learners.values()}),
        "written": str(OUT / "panel_dev200.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
