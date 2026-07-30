#!/usr/bin/env python3
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="promotion/ronpo-gemma2-2b-uf5-anneal-s42",
        revision="d06cff7476d54571ed76e6ce606360176ff31c24",
        allow_patterns=["stronger_signal/stage4/*"],
        local_dir=a.output,
    )
    model = a.output / "stronger_signal" / "stage4"
    required = ("config.json", "model.safetensors", "tokenizer_config.json")
    missing = [x for x in required if not (model / x).is_file()]
    if missing:
        raise RuntimeError(f"incomplete parent download: {missing}")
    print(model)


if __name__ == "__main__":
    main()
