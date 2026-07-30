#!/usr/bin/env python3
"""Fail-closed identity and integrity checks for downloaded screen models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "llama31": {"model_type": "llama", "layers": 32, "hidden": 4096, "official_repo": "meta-llama/Llama-3.1-8B-Instruct"},
    "qwen25": {"model_type": "qwen2", "layers": 28, "hidden": 3584, "official_repo": "Qwen/Qwen2.5-7B-Instruct"},
    "mistral7": {"model_type": "mistral", "layers": 32, "hidden": 4096, "official_repo": "mistralai/Mistral-7B-Instruct-v0.3"},
    "zephyr": {"model_type": "mistral", "layers": 32, "hidden": 4096, "official_repo": "HuggingFaceH4/zephyr-7b-beta"},
    "wildguard": {"model_type": "mistral", "layers": 32, "hidden": 4096, "official_repo": "allenai/wildguard"},
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    downloads = args.root / "downloads"
    result = {"status": "complete", "models": {}}
    failed = []
    for name, expected in EXPECTED.items():
        path_file = downloads / f"{name}.path"
        local_dir = Path(path_file.read_text(encoding="utf-8").strip())
        config_path = local_dir / "config.json"
        tokenizer_path = local_dir / "tokenizer_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        tensors = sorted(local_dir.glob("*.safetensors"))
        checks = {
            "config_exists": config_path.is_file(),
            "tokenizer_config_exists": tokenizer_path.is_file(),
            "model_type": config.get("model_type") == expected["model_type"],
            "num_hidden_layers": config.get("num_hidden_layers") == expected["layers"],
            "hidden_size": config.get("hidden_size") == expected["hidden"],
            "unquantized": "quantization_config" not in config,
            # WildGuard is evaluated through its published classifier prompt,
            # not a chat template. The three generator bases and Zephyr must
            # expose one because decode uses each model's native template.
            "chat_template_present": name == "wildguard" or bool(tokenizer.get("chat_template")),
            "safetensor_bytes_gt_12GB": sum(path.stat().st_size for path in tensors) > 12_000_000_000,
        }
        passed = all(checks.values())
        if not passed:
            failed.append(name)
        status = (downloads / f"{name}.status").read_text(encoding="utf-8").strip()
        result["models"][name] = {
            "official_repo": expected["official_repo"],
            "local_dir": str(local_dir),
            "status_line": status,
            "config_sha256": digest(config_path),
            "tokenizer_config_sha256": digest(tokenizer_path),
            "safetensor_count": len(tensors),
            "safetensor_bytes": sum(path.stat().st_size for path in tensors),
            "safetensor_files": [
                {"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)} for path in tensors
            ],
            "checks": checks,
            "passed": passed,
        }
    result["all_passed"] = not failed
    result["failed_models"] = failed
    output = args.root / "model_artifact_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_passed": result["all_passed"], "failed_models": failed}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
