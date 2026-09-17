"""Fetch the two pinned public checkpoints the UF-4 panel needs.

Public model weights only. No paid inference API is contacted here, and nothing
is uploaded: this is a read of two revision-pinned Hugging Face repositories.
"""
import json, os, time

os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ["HF_HOME"] = "/work/hf_cache"

from huggingface_hub import snapshot_download

TARGETS = [
    ("answerdotai/ModernBERT-base", "8949b909ec900327062f0ebf497f51aef5e6f0c8",
     "/work/uf4_20260910/assets/ModernBERT-base"),
    ("Qwen/Qwen3-14B", "40c069824f4251a91eefaf281ebe4c544efd3e18",
     "/work/uf4_20260910/assets/Qwen3-14B"),
]

report = {}
for repo, revision, local in TARGETS:
    t0 = time.time()
    path = snapshot_download(repo, revision=revision, local_dir=local,
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.md",
                                             "*.model", "*.py"],
                             max_workers=8)
    total = sum(f.stat().st_size for f in __import__("pathlib").Path(path).rglob("*") if f.is_file())
    report[repo] = {"revision": revision, "path": path,
                    "seconds": round(time.time() - t0, 1),
                    "bytes": total}
    print(json.dumps({repo: report[repo]}), flush=True)

with open("/work/uf4_20260910/provenance/downloaded_assets.json", "w") as f:
    json.dump(report, f, indent=2)
print("ALL_DONE", flush=True)
