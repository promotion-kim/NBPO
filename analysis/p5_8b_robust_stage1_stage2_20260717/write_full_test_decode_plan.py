#!/usr/bin/env python3
"""Write the fixed checkpoint-to-worker assignment before retrospective decode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    exp = Path(args.experiment)
    paths = {"base": args.base}
    for arm in [
        "ronpo_os_stage2", "ronpo_topmass_stage2", "inpo_avg_stage2", "simpo_stage2", "ipo_stage2",
        "dpo_stage2", "sppo_avg_stage2", "ht_mnpo_harmless_stage2", "ht_mnpo_helpfulness_stage2",
    ]:
        paths[arm] = str(exp / "stage2" / arm / "train" / "full")
    payload = {
        "status": "locked_before_decode",
        "scope": "retrospective diagnostic only",
        "models": paths,
        "workers": {
            "aiprlab_ronpo_gpu_0_3": ["base", "ronpo_os_stage2", "ronpo_topmass_stage2", "inpo_avg_stage2", "dpo_stage2", "sppo_avg_stage2"],
            "aiprlab_ronpo2_gpu_0_1": ["simpo_stage2", "ipo_stage2", "ht_mnpo_harmless_stage2", "ht_mnpo_helpfulness_stage2"],
        },
        "spent_sealed_split_touched": False,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
