"""Materialize the 500-prompt cross-play split and queue generation for the fixed bank.

Three of the four bank policies are fixed by the declaration and do not depend on
competitor selection: the base, NBPO seed 42 and fixed-reference Nash seed 42.
Their fresh responses can therefore be generated now, and generating them now is
useful beyond getting ahead: each is a single-GPU job of about nine minutes,
which is exactly the shape that fills the 24-minute window where a judge holds
one card and the controller reserves the other three for the next 4-GPU
training. That window currently goes unused once per arm, because the
arc/hellaswag/mmlu backfill is exhausted.

Sampling is the pool generator unchanged -- same 8 learner and 8 comparator
occurrences, temperature 1.0, top_p 1.0, same length limits -- so cross-play
responses are drawn under the identical contract as the training pool and the
final evaluation, on prompts none of those ever touched.
"""
import json, pathlib
R = pathlib.Path("/work/uf4_20260910")
Q = R / "jobs/queue"
SPLIT_DIR = R / "splits/v1_reproduce"

selected = json.loads((SPLIT_DIR / "crossplay_500.json").read_text())
wanted = set(selected["prompt_ids"])
rows = [json.loads(l) for l in (SPLIT_DIR / "unassigned.jsonl").open() if l.strip()]
keep = [r for r in rows if r["prompt_id"] in wanted]
if len(keep) != len(wanted):
    raise SystemExit("selected ids not all present: %d of %d" % (len(keep), len(wanted)))
# emit in the declared order (the crossplay_500.json id order)
order = {pid: i for i, pid in enumerate(selected["prompt_ids"])}
keep.sort(key=lambda r: order[r["prompt_id"]])
path = SPLIT_DIR / "crossplay_500.jsonl"
if not path.exists():
    with path.open("x") as stream:
        for r in keep:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", path, len(keep), "prompts")
else:
    print("split already present:", path)

GEN_ENV = {"PYTHONPATH": "/work/pylibs_eval:/work/uf4_20260910/code",
           "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
           "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "OMP_NUM_THREADS": "4",
           "WANDB_MODE": "disabled"}
BANK = [("base", "/work/models/bases/Llama-3.1-8B-Instruct"),
        ("nbpo_mse_s42", str(R / "arms/nbpo_mse_s42")),
        ("fixedref_mse_s42", str(R / "arms/fixedref_mse_s42"))]

for name, model in BANK:
    spec = {"job_id": "uf4_crossplay_gen_%s" % name, "priority": 64, "gpus": 1,
            "cwd": str(R), "env": GEN_ENV, "depends_on": [],
            "command": ["python3", str(R / "code/generate_uf4_pool.py"),
                        "--splits", "v1_reproduce", "--split-names", "crossplay_500",
                        "--out-name", "crossplay_%s" % name, "--model", model,
                        "--shard", "0", "--shards", "1", "--prompts-per-chunk", "25"],
            "timeout_s": 21600,
            "artifacts": [str(R / ("pools/crossplay_%s/shard0/complete_shard0.json" % name))]}
    p = Q / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
    payload = json.dumps(spec, indent=2) + "\n"
    if p.exists() and p.read_text() != payload:
        raise SystemExit("refusing to change existing spec: %s" % p)
    if p.exists():
        print("unchanged", spec["job_id"]); continue
    p.write_text(payload)
    print("queued %-36s prio %d gpus %d" % (spec["job_id"], spec["priority"], spec["gpus"]))
