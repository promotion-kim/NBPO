#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path


ROOT = Path("/NHNHOME/AIPR/sjkim/ronpo_uf5_anneal_20260722").resolve()


def bytes_used(path):
    if path.is_symlink():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(x.stat().st_size for x in path.rglob("*") if x.is_file() and not x.is_symlink())


def referenced_by_process(path):
    needle = str(path).encode()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if needle in (proc / "cmdline").read_bytes():
                return proc.name
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return None


def remove_exact(path):
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    a = p.parse_args()
    root = a.root.resolve()
    if root != ROOT or not (root / "SCORING_COMPLETE").is_file():
        raise RuntimeError("pruning requires the completed, exact new experiment root")

    audits = []
    for path in sorted((root / "hf_uploads").glob("*.json")):
        row = json.loads(path.read_text())
        if not (row.get("verified") and row.get("public") and row.get("fresh_remote_config_load")
                and row.get("fresh_remote_tokenizer_load")
                and row.get("local_weight_sha256") == row.get("remote_lfs_weight_sha256")):
            raise RuntimeError(f"unverified upload audit: {path}")
        local = Path(row["local_path"]).resolve()
        if root not in local.parents:
            raise RuntimeError(f"checkpoint outside approved experiment namespace: {local}")
        audits.append((path, row, local))

    expected = {
        root / "hf_uploads" / f"{arm}_stage{stage}.json"
        for arm in ("moving_anchor", "stronger_signal")
        for stage in range(1, 5)
        if (root / arm / f"stage{stage}" / "STAGE_COMPLETE").is_file()
    }
    found = {x[0] for x in audits}
    if expected != found:
        raise RuntimeError(f"upload ledger mismatch: missing={sorted(map(str, expected-found))}")

    ledger, total = [], 0
    for audit_path, row, local in audits:
        candidates = [local / "model.safetensors"]
        candidates += sorted(local.glob("checkpoint-*"))
        for path in candidates:
            if not path.exists():
                continue
            owner = path.stat().st_uid
            if owner != os.getuid():
                raise RuntimeError(f"ownership mismatch: {path}")
            pid = referenced_by_process(path)
            if pid:
                raise RuntimeError(f"active process {pid} references {path}")
            size = bytes_used(path)
            remove_exact(path)
            total += size
            ledger.append({
                "method": "RONPO objective-stratified", "arm": row["path_in_repo"].split("/")[0],
                "stage": int(row["path_in_repo"].split("stage")[-1]), "local_path": str(path),
                "bytes_freed": size, "hf_repo": row["repo"], "revision": row["verified_revision"],
                "verified": True, "action": "deleted verified redundant local model artifact",
            })

    data_dirs = []
    for arm in ("moving_anchor", "stronger_signal"):
        for stage in range(1, 5):
            work = root / arm / f"stage{stage}"
            if not (work / "STAGE_COMPLETE").is_file():
                continue
            data_dirs += [work / x for x in ("pool", "scored", "pairs", "precomputed_raw", "precomputed_os")]
    data_dirs.append(root / "stronger_signal" / "shared_base")
    for path in data_dirs:
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink():
            # Remove only the link inside the new run; never follow the locked Stage-1 target.
            ledger.append({"local_path": str(path), "bytes_freed": 0, "action": "removed local symlink only"})
            path.unlink()
            continue
        resolved = path.resolve()
        if root not in resolved.parents:
            raise RuntimeError(f"intermediate path escapes experiment root: {resolved}")
        pid = referenced_by_process(resolved)
        if pid:
            raise RuntimeError(f"active process {pid} references {resolved}")
        size = bytes_used(resolved)
        remove_exact(resolved)
        total += size
        ledger.append({"local_path": str(resolved), "bytes_freed": size,
                       "action": "deleted completed intermediate; scores/logs/eval/audit retained"})

    payload = {"total_bytes_freed": total, "entries": ledger}
    (root / "RETENTION_LEDGER.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Verified upload and pruning ledger", "", f"Freed {total} bytes.", ""]
    lines += [f"- `{x['local_path']}`: {x['action']} ({x['bytes_freed']} bytes)" for x in ledger]
    (root / "PRUNE_LOG.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
