#!/usr/bin/env python3
"""Materialize the preregistered policy decode queues."""

import json
from pathlib import Path

ROOT = "/NHNHOME/AIPR/sjkim"
REPO = f"{ROOT}/MNPO_rev_20260720"
PYTHON = f"{ROOT}/venv_clean/bin/python"
WRAPPER = f"{ROOT}/novelty_defense_20260723/code/run_decode_wandb.py"
DATA = f"{REPO}/data/gemma2_ufb_part2_test.jsonl"
OUTPUT = f"{ROOT}/novelty_defense_20260723/generations"

MODELS = {
    "base": "Qwen/Qwen2.5-1.5B-Instruct",
    "ht_skywork": "promotion/htmnpo-skywork-qwen25-1p5b-stage2",
    "ht_athene": "promotion/htmnpo-athene-qwen25-1p5b-stage2",
    "ht_armo": "promotion/htmnpo-armorm-qwen25-1p5b-stage2",
    "ronpo_os": f"{ROOT}/ronpo_arms_20260720/os-ronpo-os-k05-b200",
    "ronpo_topmass": f"{ROOT}/novelty_defense_20260723/checkpoints/topmass_s42",
    "ronpo_konly": f"{ROOT}/ronpo_arms_20260720/os-ronpo-konly-k05-b200",
    "ronpo_aonly": f"{ROOT}/ronpo_arms_20260720/os-ronpo-aonly-k05-b200",
    "sppo_avg_s2": f"{ROOT}/avg_s2_20260720/out/qwen2.5-1.5b-instruct_sppo_avg_online_multiobj_stage_2",
    "inpo_avg_s2": f"{ROOT}/avg_s2_20260720/out/qwen2.5-1.5b-instruct_inpo_avg_online_multiobj_stage_2",
    "maxmin_rlhf": f"{ROOT}/avg_s2_20260720/out/qwen2.5-1.5b-instruct_inpo_maxmin_online_multiobj_stage_2",
    "ronpo_lam4": f"{ROOT}/ronpo_arms_20260720/os-ronpo-os-k05-lam4",
    "ronpo_lam16": f"{ROOT}/ronpo_arms_20260720/os-ronpo-os-k05-lam16",
    "ronpo_os_s43": f"{ROOT}/ronpo_arms_20260720/os-ronpo-os-k05-s43",
    "ronpo_konly_s43": f"{ROOT}/ronpo_arms_20260720/os-ronpo-konly-k05-s43",
}


def job(name: str) -> dict:
    return {
        "name": name,
        "command": [
            PYTHON, WRAPPER,
            "--name", name,
            "--model", MODELS[name],
            "--data", DATA,
            "--output", f"{OUTPUT}/{name}",
            "--python", PYTHON,
            "--repo", REPO,
        ],
    }


def main() -> None:
    output = Path("results/novelty_defense_20260723/decode_queues")
    output.mkdir(parents=True, exist_ok=True)
    ronpo = [
        "base", "ht_skywork", "ht_athene", "ht_armo", "sppo_avg_s2",
        "inpo_avg_s2", "maxmin_rlhf", "ronpo_lam4", "ronpo_lam16",
        "ronpo_topmass",
    ]
    ronpo2 = ["ronpo_os", "ronpo_konly", "ronpo_aonly", "ronpo_os_s43", "ronpo_konly_s43"]
    (output / "ronpo.json").write_text(json.dumps([job(x) for x in ronpo], indent=2) + "\n")
    (output / "ronpo2.json").write_text(json.dumps([job(x) for x in ronpo2], indent=2) + "\n")


if __name__ == "__main__":
    main()
