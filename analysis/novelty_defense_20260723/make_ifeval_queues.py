#!/usr/bin/env python3
"""Build fixed IFEval queues for policies lacking exact verified reports."""

import json
from pathlib import Path

ROOT = "/NHNHOME/AIPR/sjkim"
RUNNER = f"{ROOT}/novelty_defense_20260723/code/run_ifeval_one.sh"
MODELS = {
    "ronpo_os": f"{ROOT}/ronpo_arms_20260720/os-ronpo-os-k05-b200",
    "ronpo_topmass": f"{ROOT}/novelty_defense_20260723/checkpoints/topmass_s42",
    "ronpo_konly": f"{ROOT}/ronpo_arms_20260720/os-ronpo-konly-k05-b200",
    "ronpo_aonly": f"{ROOT}/ronpo_arms_20260720/os-ronpo-aonly-k05-b200",
    "sppo_avg_s2": f"{ROOT}/avg_s2_20260720/out/qwen2.5-1.5b-instruct_sppo_avg_online_multiobj_stage_2",
    "inpo_avg_s2": f"{ROOT}/avg_s2_20260720/out/qwen2.5-1.5b-instruct_inpo_avg_online_multiobj_stage_2",
    "maxmin_rlhf": f"{ROOT}/avg_s2_20260720/out/qwen2.5-1.5b-instruct_inpo_maxmin_online_multiobj_stage_2",
    "ronpo_lam4": f"{ROOT}/ronpo_arms_20260720/os-ronpo-os-k05-lam4",
    "ronpo_lam16": f"{ROOT}/ronpo_arms_20260720/os-ronpo-os-k05-lam16",
}


def job(name: str, port: int) -> dict:
    return {"name": name, "command": ["bash", RUNNER, name, MODELS[name], str(port)]}


def main() -> None:
    output = Path("results/novelty_defense_20260723/ifeval_queues")
    output.mkdir(parents=True, exist_ok=True)
    queues = {
        "ronpo_gpu3": ["ronpo_konly"],
        "ronpo2_gpu0": ["ronpo_aonly", "sppo_avg_s2", "inpo_avg_s2"],
        "ronpo2_gpu1": ["maxmin_rlhf", "ronpo_lam4", "ronpo_lam16"],
        "ronpo_gpu1_extra": ["sppo_avg_s2", "ronpo_lam4"],
        "ronpo_gpu2_extra": ["inpo_avg_s2", "ronpo_lam16"],
        "ronpo_gpu3_lams": ["ronpo_lam4", "ronpo_lam16"],
    }
    for qi, (name, policies) in enumerate(queues.items()):
        (output / f"{name}.json").write_text(json.dumps([job(p, 9300 + qi) for p in policies], indent=2) + "\n")


if __name__ == "__main__":
    main()
