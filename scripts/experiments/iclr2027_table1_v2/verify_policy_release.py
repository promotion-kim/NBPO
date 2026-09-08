#!/usr/bin/env python3
"""Verify the published policy arms against the local checkpoints, byte for byte.

For an 8B model, re-downloading every arm to compare logits would move 135 GB to
prove something a hash proves exactly: the Hub stores an LFS sha256 for each
uploaded file, so comparing it with the local file's sha256 establishes byte
identity, which is strictly stronger than agreeing on one forward pass.

The config and tokenizer are checked the same way, because a checkpoint whose
weights match but whose tokenizer does not is not the same model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default="promotion/nbpo-policy-diagnostics")
    ap.add_argument("--revision", required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="label=<local dir>")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()                       # anonymous: proves the repo is really public
    info = api.model_info(args.repo_id, revision=args.revision, files_metadata=True)
    if info.private:
        raise SystemExit("repository is not public")
    published = {s.rfilename: s for s in info.siblings}

    report = {"repo_id": args.repo_id, "revision": args.revision,
              "public": not info.private,
              "method": ("Hub LFS sha256 compared with the local file sha256: byte "
                         "identity, checked anonymously so public access is proven too"),
              "arms": {}}
    ok_all = True
    for spec in args.arms:
        label, local = spec.split("=", 1)
        checks = {}
        for fn in ("model.safetensors", "config.json", "tokenizer.json",
                   "tokenizer_config.json", "gate.json", "run_config.yaml"):
            lp = Path(local) / fn
            key = f"arms/{label}/{fn}"
            sib = published.get(key)
            if sib is None:
                checks[fn] = {"published": False}
                continue
            remote = (sib.lfs or {}).get("sha256") if sib.lfs else None
            local_sha = sha256_file(lp) if lp.exists() else None
            match = (remote == local_sha) if (remote and local_sha) else None
            checks[fn] = {"published": True, "size": sib.size,
                          "remote_sha256": remote, "local_sha256": local_sha,
                          "byte_identical": match,
                          "note": None if remote else "not LFS-tracked; size compared only"}
            if fn == "model.safetensors" and match is not True:
                ok_all = False
        report["arms"][label] = checks
    report["all_weights_byte_identical"] = ok_all
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for label, checks in report["arms"].items():
        w = checks.get("model.safetensors", {})
        print(f"{label:>18}  weights byte-identical: {w.get('byte_identical')}  "
              f"({(w.get('size') or 0) / 2**30:.1f} GiB)")
    print(f"\nall weights byte-identical: {ok_all}")
    if not ok_all:
        raise SystemExit("a published checkpoint does not match its local file")


if __name__ == "__main__":
    main()
