"""Run explicitly on nbpo-judge; install only into the authorized task directory.

Reads the pre-reviewed pip dry-run plan, installs only its missing noncritical
packages without dependency resolution, and verifies official HarmBench CPU
helper routing. Never launches a model or a GPU job.
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.parse

ROOT = Path("/work/nbpo_repair_20260909")
DEPS = ROOT / "deps_harmbench"
SITE = DEPS / "site"
REPO = ROOT / "external/HarmBench"
COMMIT = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"
MODEL_URL = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
CRITICAL = ("numpy", "pydantic", "torch", "transformers", "vllm")


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(data)
    return digest.hexdigest()


def import_versions():
    result = {}
    for name in CRITICAL:
        module = importlib.import_module(name)
        result[name] = {"version": importlib.metadata.version(name), "file": module.__file__}
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU helper setup unexpectedly initialized CUDA")
    return result


def install():
    plan = json.loads((DEPS / "install_plan_pinned.json").read_text())
    urls = []
    forbidden = set(CRITICAL) | {"pydantic-core", "scipy", "tokenizers", "accelerate", "deepspeed", "typer"}
    for item in plan["install"]:
        name = item["metadata"]["name"].lower().replace("_", "-")
        if name in forbidden:
            raise ValueError(f"Refusing critical/shared dependency replacement: {name}")
        url = item["download_info"]["url"]
        if urllib.parse.urlparse(url).hostname != "files.pythonhosted.org":
            raise ValueError("Only official PyPI wheel-host URLs are permitted")
        sha = item["download_info"]["archive_info"]["hashes"]["sha256"]
        urls.append(f"{url}#sha256={sha}")
    baseline = import_versions()
    write_json(DEPS / "critical_imports_before.json", baseline)
    environment = dict(os.environ, TMPDIR=str(DEPS / "tmp"), PIP_DISABLE_PIP_VERSION_CHECK="1")
    command = [sys.executable, "-m", "pip", "install", "--no-deps", "--no-cache-dir",
               "--target", str(SITE), "--report", str(DEPS / "installed_packages.json"), *urls]
    with (DEPS / "install.log").open("x") as log:
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    command = [sys.executable, "-m", "pip", "install", "--no-deps", "--no-cache-dir",
               "--target", str(SITE), "--report", str(DEPS / "installed_model.json"), MODEL_URL]
    with (DEPS / "install_model.log").open("x") as log:
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("Task-local package and official model installs completed; run verify in a fresh process.")


def verify():
    baseline = json.loads((DEPS / "critical_imports_before.json").read_text())
    after = import_versions()
    if after != baseline:
        raise ValueError(f"Critical imports changed: before={baseline}, after={after}")
    from scripts.experiments.nbpo_repair_20260909.evaluate_responses import load_harmbench_helpers, harmbench_route
    helper = load_harmbench_helpers(REPO)
    if not callable(helper.compute_results_classifier):
        raise TypeError("Official classifier helper failed import")
    with (REPO / "data/behavior_datasets/harmbench_behaviors_text_test.csv").open() as stream:
        behaviors = list(csv.DictReader(stream))
    probes = []
    previous = Path.cwd()
    try:
        os.chdir(REPO)
        for kind in ("book", "lyrics"):
            behavior = next(b for b in behaviors if "hash_check" in b["Tags"] and kind in b["Tags"].split(", "))
            if harmbench_route(behavior) != "hash_check":
                raise ValueError("Copyright behavior misrouted to classifier")
            benign = "This is a harmless pipeline verification sentence about checking software. " * 100
            result = helper.compute_results_hashing(behavior, [{"generation": benign}])
            if len(result) != 1 or result[0]["label"] != 0:
                raise ValueError("Benign hash-route probe did not produce its expected negative label")
            probes.append({"kind": kind, "behavior_id": behavior["BehaviorID"], "label": 0,
                           "generation_sha256": hashlib.sha256(benign.encode()).hexdigest()})
    finally:
        os.chdir(previous)
    # No targets or generated passages are copied into this report.
    packages = []
    for filename in ("installed_packages.json", "installed_model.json"):
        report = json.loads((DEPS / filename).read_text())
        packages.extend({"name": item["metadata"]["name"], "version": item["metadata"]["version"],
            "source_url": item["download_info"]["url"],
            "wheel_sha256": item["download_info"]["archive_info"]["hashes"]["sha256"]}
            for item in report["install"])
    tracked = subprocess.check_output(["git", "-C", str(REPO), "ls-files", "eval_utils.py",
        "data/behavior_datasets/harmbench_behaviors_text_test.csv", "data/copyright_classifier_hashes"], text=True).splitlines()
    tracked_hashes = {path: file_hash(REPO / path) for path in tracked}
    result = {"status": "ready_for_root_assigned_gpu_scoring", "official_repo": "https://github.com/centerforaisafety/HarmBench",
        "source_commit": COMMIT, "tracked_helper_and_data_sha256": tracked_hashes,
        "packages": packages, "critical_imports_before": baseline, "critical_imports_after": after,
        "critical_imports_unchanged": True, "hash_route_probes": probes,
        "classifier_helper_imported": True, "gpu_model_launched": False,
        "runtime_pythonpath_prefix": str(SITE), "official_spacy_model_source": MODEL_URL,
        "global_packages_installed_or_modified": False}
    write_json(DEPS / "runtime_validation.json", result)
    print(json.dumps({"status": result["status"], "package_count": len(packages), "critical_imports_unchanged": True,
                      "hash_route_probes": probes, "tracked_files": len(tracked_hashes)}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("install", "verify"):
        raise SystemExit("Usage: setup_harmbench_runtime.py install|verify")
    globals()[sys.argv[1]]()
