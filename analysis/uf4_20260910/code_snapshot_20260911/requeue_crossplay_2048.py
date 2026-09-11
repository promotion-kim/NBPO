"""Third attempt at the cross-play token budget, and a fix to what it exposed.

Parse rate by budget on these inputs: 3% at 256, 82% at 1024, and the failures
are still cut off mid-reasoning rather than finishing without a verdict. It is
also rubric-dependent -- helpfulness 92.4%, truthfulness 86.0%, instruction
following 76.2%, honesty 74.5% -- which is what a length limit looks like when
some comparisons need more deliberation than others. 2048 is the next step.

The final evaluation reaches 99.5% at 256 on the same judge with a
byte-identical template and statistically identical prompt and response lengths,
and I have not explained that gap. It is recorded as unexplained rather than
papered over: the budget is being raised because it demonstrably raises the
parse rate, not because the difference is understood.

The 1024-token outputs are kept alongside the 256-token ones so the progression
stays inspectable.
"""
import json, pathlib, shutil
R = pathlib.Path("/work/uf4_20260910")
Q = R / "jobs/queue"
PAIRS = R / "evaluation/crossplay/pairs"
ENV = {"PYTHONPATH": "/work/pylibs_eval:/work/uf4_20260910/code",
       "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
       "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "OMP_NUM_THREADS": "4",
       "WANDB_MODE": "disabled"}
TAGS = [("base", "fixedref_mse_s42"), ("base", "nbpo_mse_s42"),
        ("fixedref_mse_s42", "nbpo_mse_s42")]

for a, b in TAGS:
    tag = "%s__vs__%s" % (a, b)
    live = PAIRS / tag
    if live.exists():
        aside = PAIRS / (tag + ".max_tokens1024_parse82pct")
        if aside.exists():
            shutil.rmtree(aside)
        live.rename(aside)
        print("kept aside", aside.name)
    spec = {"job_id": "uf4_crossplay_judge_%s_t2048" % tag, "priority": 64, "gpus": 1,
            "cwd": str(R), "env": ENV, "depends_on": [],
            "command": ["python3", str(R / "code/judge_crossplay_pair.py"),
                        "--policy-a", a, "--policy-b", b,
                        "--pool-a", str(R / ("pools/crossplay_%s" % a)),
                        "--pool-b", str(R / ("pools/crossplay_%s" % b)),
                        "--max-tokens", "2048"],
            "timeout_s": 43200,
            "artifacts": [str(PAIRS / tag / "complete.json")]}
    p = Q / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
    payload = json.dumps(spec, indent=2) + "\n"
    if p.exists() and p.read_text() != payload:
        raise SystemExit("refusing to change existing spec: %s" % p)
    if p.exists():
        print("unchanged", spec["job_id"]); continue
    p.write_text(payload)
    print("queued", spec["job_id"])

for path in sorted(Q.glob("*.json")):
    spec = json.loads(path.read_text())
    if spec["job_id"] != "uf4_crossplay_aggregate":
        continue
    spec["depends_on"] = ["uf4_crossplay_judge_%s__vs__%s_t2048" % t for t in TAGS]
    spec["retry"] = int(spec.get("retry", 0)) + 1
    path.write_text(json.dumps(spec, indent=2) + "\n")
    print("aggregator now waits on the 2048-token judges")
