"""Test-split rubric map and the evaluation job specs for one panel.

The generation step draws ONE response per test prompt from each arm at the
training pool's decoding, with a per-arm seed namespace so the base's own row is
an independent draw and not a byte copy of the comparator. The comparator is a
separate base draw under its own namespace, which is what makes the "base fresh
draw" row a genuine null rather than a self-comparison.
"""
import argparse, hashlib, json
from pathlib import Path

UF = Path("/work/uf4_20260910")
SUB = Path("/work/sub_20260914")
BASE_MODEL = "/work/models/bases/Qwen2.5-7B-Instruct"
BASE_REV = "a09a35458c702b33eeacc393d103063234e8bc28"

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--panel", required=True)
ap.add_argument("--panel-tag", required=True, help="e.g. us_v1")
ap.add_argument("--arms", nargs="+", required=True)
ap.add_argument("--max-tokens", type=int, default=1024)
ap.add_argument("--judge-max-tokens", type=int, default=1024)
args = ap.parse_args()

p = args.panel
panel_dir = SUB / "panel" / args.panel_tag
freeze = json.loads((panel_dir / "freeze.json").read_text())
objs = freeze["objectives"]

items = {}
for line in (panel_dir / "test.jsonl").open():
    if line.strip():
        r = json.loads(line)
        items[r["prompt_id"]] = [
            {"item_index": i, "text": o["text"], "importance": None,
             "objective_id": o["objective_id"]} for i, o in enumerate(objs)]
ip = panel_dir / "test_items.json"
ip.write_text(json.dumps(items, ensure_ascii=False))

QD = UF / "jobs/queue"
GEN_ENV = {"PYTHONPATH": "/work/pylibs_eval:/work/sub_20260914/code",
           "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
           "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "OMP_NUM_THREADS": "4",
           "WANDB_MODE": "disabled"}

# every row of the table, plus the comparator draw
draws = [("comparator", BASE_MODEL, "%s_eval_comparator" % p),
         ("base", BASE_MODEL, "%s_eval_base" % p)]
for arm in args.arms:
    draws.append((arm, str(UF / "arms" / ("%s_%s" % (p, arm))), "%s_eval_%s" % (p, arm)))

written, gen_ids = [], []
for name, model, seed_tag in draws:
    jid = "%s_egen_%s" % (p, name)
    spec = {
        "job_id": jid, "priority": 130, "gpus": 1,
        "depends_on": ([] if name in ("comparator", "base")
                       else ["%s_train_%s" % (p, name)]),
        "cwd": "/work/sub_20260914/code", "env": dict(GEN_ENV), "timeout_s": 21600,
        "command": ["python3", "/work/sub_20260914/code/gen_candidates.py",
                    "--panel", "%s/test.jsonl" % args.panel_tag,
                    "--model", model, "--model-revision", BASE_REV,
                    "--responses-per-prompt", "1",
                    "--temperature", "1.0", "--top-p", "1.0",
                    "--max-tokens", str(args.max_tokens),
                    "--max-model-len", "4096",
                    "--seed-tag", seed_tag,
                    "--tag", "%seval_%s" % ("%s_" % p, name)],
        "artifacts": [str(SUB / "responses" / ("%s_eval_%s" % (p, name)) / "responses.jsonl")],
        "note": ("one draw per test prompt at the training pool decoding; seed "
                 "namespace %s keeps this draw independent of every other" % seed_tag),
    }
    (QD / ("130_%s_egen_%s.json" % (p, name))).write_text(json.dumps(spec, indent=1) + "\n")
    written.append(jid); gen_ids.append(jid)

judge = {
    "job_id": "%s_ejudge" % p, "priority": 131, "gpus": 4,
    "depends_on": list(gen_ids), "cwd": "/work/sub_20260914/code",
    "env": {"PYTHONPATH": "/work/pylibs_eval:/work/sub_20260914/code",
            "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "OMP_NUM_THREADS": "4",
            "WANDB_MODE": "disabled"},
    "timeout_s": 86400,
    "command": ["python3", "/work/sub_20260914/code/panel_eval_judge.py",
                "--bank", "base"] + list(args.arms) + [
                "--comparator", "comparator",
                "--prefix", "%s_eval_" % p,
                "--items", str(ip),
                "--out-tag", "%s_objwise" % p,
                "--judge", "/work/models/bases/Qwen2.5-72B-Instruct",
                "--judge-revision", "495f39366efef23836d0cfae4fbe635880d2be31",
                "--tensor-parallel-size", "4",
                "--temperature", "0.0",
                "--max-tokens", str(args.judge_max_tokens),
                "--max-model-len", "32768",
                "--chunk", "4000"],
    "artifacts": [str(SUB / "objectivewise" / ("%s_objwise" % p) / "complete.json")],
    "note": ("objective-wise wins against one fixed base draw, both orders, the "
             "training judge's PSC template under the independent 72B evaluator"),
}
(QD / ("131_%s_ejudge.json" % p)).write_text(json.dumps(judge, indent=1) + "\n")
written.append(judge["job_id"])

print(json.dumps({"panel": p, "test_prompts": len(items),
                  "objectives": len(objs),
                  "items_sha256": hashlib.sha256(ip.read_bytes()).hexdigest()[:16],
                  "queued": written,
                  "planned_verdicts": (1 + len(args.arms)) * len(items) * len(objs) * 2},
                 indent=1))
