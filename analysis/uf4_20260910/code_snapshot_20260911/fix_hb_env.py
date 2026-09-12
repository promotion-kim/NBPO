"""Give the HarmBench scoring job the PYTHONPATH the helper actually needs.

HarmBench's eval_utils.py imports spaCy at module scope, and spaCy is not in
/work/pylibs_eval -- it lives in a dedicated site directory that the previous
campaign recorded for exactly this purpose:

    deps_harmbench/runtime_validation.json
      runtime_pythonpath_prefix = "/work/nbpo_repair_20260909/deps_harmbench/site"
      status = "ready_for_root_assigned_gpu_scoring"
      classifier_helper_imported = true

Verified before editing: with that prefix prepended, eval_utils.py imports and
en_core_web_sm loads. Without it the scoring job dies on ModuleNotFoundError
after the ten generations have already run.

The generation jobs are untouched; they never import the helper.
"""
import json, pathlib
Q = pathlib.Path("/work/uf4_20260910/jobs/queue")
PREFIX = "/work/nbpo_repair_20260909/deps_harmbench/site"
for path in sorted(Q.glob("*.json")):
    spec = json.loads(path.read_text())
    if spec["job_id"] != "uf4_score_harmbench2":
        continue
    old = spec["env"]["PYTHONPATH"]
    if old.startswith(PREFIX):
        print("already prefixed"); break
    spec["env"]["PYTHONPATH"] = PREFIX + ":" + old
    spec["env_note"] = ("spaCy for HarmBench's eval_utils lives in deps_harmbench/site, "
                        "per runtime_validation.json's runtime_pythonpath_prefix")
    path.write_text(json.dumps(spec, indent=2) + "\n")
    print("PYTHONPATH now:", spec["env"]["PYTHONPATH"])
    break
