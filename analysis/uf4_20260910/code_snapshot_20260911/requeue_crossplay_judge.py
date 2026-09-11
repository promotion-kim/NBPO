"""Re-run the cross-play judging with a token budget the judge can finish in.

At max_tokens 256 the three pairs parsed 127, 105 and 275 verdicts out of 4,000
-- about 3% -- and the unparsed outputs are visibly cut off mid-reasoning. The
same judge at the same budget parses 99.4-99.6% on the final evaluation, and the
template, verdict grammar, prompt lengths and response lengths are all
statistically identical between the two, so the budget is marginal rather than
wrong and cross-play inputs happen to sit past the margin.

Raising it to 1024 is a protocol choice for this evaluation and is recorded as
one. It does NOT touch the final evaluation, whose numbers are already reported
at 256 with a 99.5% parse rate; changing that retroactively would alter
published cells for no measured reason.

New job ids, because the old ones are DONE and the controller never re-runs a
DONE job. The 3%-parse outputs are moved aside rather than deleted, so the
failure stays inspectable.
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
    old = PAIRS / tag
    if old.exists():
        aside = PAIRS / (tag + ".max_tokens256_parse3pct")
        if aside.exists():
            shutil.rmtree(aside)
        old.rename(aside)
        print("moved aside", aside.name)
    spec = {"job_id": "uf4_crossplay_judge_%s_t1024" % tag, "priority": 64, "gpus": 1,
            "cwd": str(R), "env": ENV, "depends_on": [],
            "command": ["python3", str(R / "code/judge_crossplay_pair.py"),
                        "--policy-a", a, "--policy-b", b,
                        "--pool-a", str(R / ("pools/crossplay_%s" % a)),
                        "--pool-b", str(R / ("pools/crossplay_%s" % b)),
                        "--max-tokens", "1024"],
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

# the aggregator must wait on the new judges, not the retired ones
for path in sorted(Q.glob("*.json")):
    spec = json.loads(path.read_text())
    if spec["job_id"] != "uf4_crossplay_aggregate":
        continue
    spec["depends_on"] = ["uf4_crossplay_judge_%s__vs__%s_t1024" % t for t in TAGS]
    spec["retry"] = int(spec.get("retry", 0)) + 1
    path.write_text(json.dumps(spec, indent=2) + "\n")
    print("aggregator now waits on the 1024-token judges")
