#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


REPO = "promotion/ronpo-gemma2-2b-uf5-anneal-s42"


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); a = p.parse_args()
    root = a.root.resolve(); approved = Path("/NHNHOME/AIPR/sjkim/ronpo_uf5_anneal_20260722").resolve()
    if root != approved:
        raise RuntimeError(f"unexpected result root: {root}")
    api = HfApi()
    api.upload_folder(repo_id=REPO, repo_type="model", folder_path=str(root / "results"),
                      path_in_repo="evaluation", commit_message="Publish common-batch evaluation")
    for local, remote in ((root / "REPORT.md", "evaluation/REPORT.md"),
                          (root / "PREREG.md", "experiment/PREREG.md"),
                          (root / "run_lock.json", "experiment/run_lock.json"),
                          (root / "fix_log.md", "experiment/fix_log.md")):
        api.upload_file(repo_id=REPO, repo_type="model", path_or_fileobj=str(local),
                        path_in_repo=remote, commit_message=f"Add {remote}")
    info = api.model_info(REPO)
    required = {"evaluation/REPORT.md", "evaluation/paired_summary.json",
                "experiment/PREREG.md", "experiment/run_lock.json", "experiment/fix_log.md"}
    files = set(api.list_repo_files(REPO, revision=info.sha))
    missing = sorted(required - files)
    if info.private or missing:
        raise RuntimeError(f"final artifact verification failed: private={info.private}, missing={missing}")
    out = root / "hf_uploads" / "results" / "final_evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"repo": REPO, "public": True, "verified_revision": info.sha,
                               "verified_files": sorted(required)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
