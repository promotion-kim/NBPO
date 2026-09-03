#!/usr/bin/env python3
"""Stub for the accelerate/run_mnpo training launch in the real-mode test.

Parses the materialized run_config.yaml with run_mnpo's own dataclasses (so a
malformed YAML fails here, like the real launcher would), then writes a
checkpoint into output_dir whose weights differ from the parent (a perturbed
copy), so its content fingerprint is new and known to the test.
"""
import argparse, json, os, shutil
from pathlib import Path
import yaml, torch
from safetensors.torch import load_file, save_file
from scripts.nbpo.run_nbpo_stage import parse_run_config

ap = argparse.ArgumentParser()
ap.add_argument("run_config")
a = ap.parse_args()
model_args, data_args, training_args = parse_run_config(a.run_config)   # real parser
cfg = yaml.safe_load(Path(a.run_config).read_text())
parent, out = Path(cfg["model_name_or_path"]), Path(cfg["output_dir"])
assert Path(list(cfg["dataset_mixer"])[0], "precompute_meta.json").exists(), "precomputed dataset missing"
out.mkdir(parents=True, exist_ok=True)
for f in parent.iterdir():
    if f.is_file() and not f.name.startswith("."):
        shutil.copy(f, out / f.name)
w = load_file(str(out / "model.safetensors"))
w = {k: v + 0.5 for k, v in w.items()}                 # "trained": weights changed, same size
save_file(w, str(out / "model.safetensors"))
(out / "MARKER.json").write_text(json.dumps({"stub_train": True, "trained_from": str(parent),
                                              "loss_type": str(training_args.loss_type)}))
print(json.dumps({"stub_train": str(out), "loss_type": str(training_args.loss_type)}))
