"""Write the two projection-pilot configs and queue them.

Both arms are the seed-42 NBPO recipe with three overrides: the 200-prompt pilot
dataset, 200 optimizer steps, and 28 rows per step (4 GPUs x 1 x 7). They share
the base checkpoint, the frozen target, the optimizer, the learning-rate
schedule, the precision and the row multiset -- 5,600 pair rows, every pair of
every prompt exactly once over the 200 steps.

The only difference is coverage per step. Arm A keeps the campaign's seeded
random sampler, so a step holds 28 pairs drawn across prompts. Arm B reads the
rows in written order, so a step holds one prompt's complete 28-pair graph,
which is the all-candidate objective evaluated exactly. This is a coverage
control at an identical token budget, and it is labelled as such rather than as
a reimplementation of the loss.
"""
import json
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
QUEUE = ROOT / "jobs" / "queue"
DIAG = ROOT / "analysis/diag_20260914"
BASE_CONFIG = ROOT / "configs/nbpo_mse_s42.yaml"
DATASET = ROOT / "datasets/diag_pilot200"
ENV = {"PYTHONPATH": "/work/nbpo_repair_20260909/deps_train:/work/nbpo_repair_20260909/code",
       "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "4",
       "MNPO_DISABLE_APEX": "1", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
       "WANDB_MODE": "disabled", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
ARMS = [("sampled", "diag_proj_sampled_s42", False, 55),
        ("all", "diag_proj_all_s42", True, 56)]


def write_queue(spec):
    existing = {json.loads(p.read_text())["job_id"] for p in QUEUE.glob("*.json")}
    if spec["job_id"] in existing:
        return "exists  " + spec["job_id"]
    path = QUEUE / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(spec, indent=2) + "\n")
    tmp.replace(path)
    return "queued  " + spec["job_id"]


def main():
    build = json.loads((DIAG / "projection/dataset_build.json").read_text())
    manifest = build["manifest_sha256"]
    lines = BASE_CONFIG.read_text().splitlines(True)

    for label, arm, sequential, priority in ARMS:
        out, seen = [], set()
        for line in lines:
            key = line.split(":", 1)[0].strip()
            if key in ("max_steps", "gradient_accumulation_steps", "eval_steps",
                       "save_steps", "output_dir", "run_name",
                       "nbpo_expected_dataset_manifest_sha256"):
                seen.add(key)
                if key == "max_steps":
                    out.append("max_steps: 200\n")
                elif key == "gradient_accumulation_steps":
                    out.append("gradient_accumulation_steps: 7\n")
                elif key == "eval_steps":
                    out.append("eval_steps: 200\n")
                elif key == "save_steps":
                    out.append("save_steps: 200\n")
                elif key == "output_dir":
                    out.append("output_dir: %s\n" % (ROOT / "arms" / arm))
                elif key == "run_name":
                    out.append("run_name: %s\n" % arm)
                else:
                    out.append("nbpo_expected_dataset_manifest_sha256: %s\n" % manifest)
            elif line.startswith("  /work/uf4_20260910/datasets/"):
                out.append("  %s: 1.0\n" % DATASET)
            else:
                out.append(line)
        if sequential:
            out.append("nbpo_sequential_sampler: true\n")
        missing = {"max_steps", "gradient_accumulation_steps", "output_dir", "run_name",
                   "nbpo_expected_dataset_manifest_sha256"} - seen
        if missing:
            raise SystemExit("base config lacks %s" % sorted(missing))
        config = ROOT / "configs" / ("%s.yaml" % arm)
        tmp = config.with_suffix(".yaml.tmp")
        tmp.write_text("".join(out))
        tmp.replace(config)
        print("wrote %s" % config)
        print(write_queue({
            "job_id": "uf4_diag_proj_%s" % label, "priority": priority, "gpus": 4,
            "cwd": "/work/nbpo_repair_20260909/code", "env": ENV, "depends_on": [],
            "command": ["python3", "-m", "torch.distributed.run", "--standalone",
                        "--nnodes=1", "--nproc_per_node=4", "-m", "mnpo_scripts.run_mnpo",
                        str(config)],
            "timeout_s": 14400,
            "artifacts": [str(ROOT / "arms" / arm / "config.json")],
        }))


if __name__ == "__main__":
    main()
