"""Capture only allowlisted run evidence, never environment secrets."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from scripts.experiments.nbpo_repair_20260909.common import file_hash, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    old = Path("/work/iclr27_table1_v2")
    files = [old / "smoke/canon/arms/canonN700_run_config.yaml",
             old / "smoke/train3/arms/DIAGrb_steps1200_run_config.yaml",
             old / "smoke/canon/precomputed/precompute_manifest.json",
             old / "smoke/train3/precomputed/precompute_manifest.json",
             old / "preserved/sr_splits/split_manifest.json",
             old / "preserved/smoke_pools_train/pool_manifest.json",
             old / "models/saferlhf_ensemble_ckpt/saferlhf_ensemble.json",
             old / "policy_release_manifest.json", old / "zero_step_probe_FIXED.json"]
    for folder in ("code", "code_reffix", "code_frozen_0131"):
        root = old / folder
        files += [p for p in root.rglob("*") if p.is_file()
                  and p.suffix in (".py", ".yaml", ".sh") and "__pycache__" not in p.parts]
    for name in ("train_canonN700.log", "train_DIAGrb_steps1200.log",
                 "precompute_canon.log", "precompute_rb.log"):
        files.append(old / "logs" / name)
    records = []
    for path in sorted(set(files)):
        if not path.is_file():
            records.append({"path": str(path), "status": "unavailable"})
            continue
        relative = path.relative_to(old)
        target = args.out / "previous_run" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        shutil.copy2(path, target)
        records.append({"path": str(path), "snapshot": str(target),
                        "sha256": file_hash(path), "bytes": path.stat().st_size})
    assets = []
    base = Path("/work/models/bases/Llama-3.1-8B-Instruct")
    paths = list(base.glob("*.json")) + list(base.glob("*.safetensors"))
    paths += list((old / "models/saferlhf_ensemble_ckpt").glob("ckpt_gpm_seed*/model.pt"))
    paths += list((old / "preserved/sr_splits").glob("*.jsonl"))
    for path in paths:
        print(f"hashing {path}", flush=True)
        assets.append({"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size})
    packages = {}
    for name in ("torch", "transformers", "tokenizers", "vllm", "accelerate", "datasets", "deepspeed", "trl", "numpy", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu", "--format=csv"], capture_output=True, text=True, check=True).stdout
    write_json(args.out / "inventory.json", {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_parent_commit": "afc527b375bab7b18f1100ece07bd18832c47508",
        "remote_git_commit": "unavailable: deployed source copies contain no .git; content hashes recorded",
        "gpu_inventory": gpu, "packages": packages, "source_files": records, "assets": assets,
        "base_revision": (base / ".cache/huggingface/download/config.json.metadata").read_text().splitlines()[0],
        "dataset_cached_revision": Path("/work/hf_cache/hub/datasets--PKU-Alignment--PKU-SafeRLHF/refs/main").read_text().strip(),
        "dataset_revision_limitation": "Cached revision alone does not prove historical extraction revision; preserved split byte hashes are authoritative.",
    })


if __name__ == "__main__":
    main()
