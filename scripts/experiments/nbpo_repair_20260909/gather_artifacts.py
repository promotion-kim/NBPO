"""Package small campaign evidence, never models, secrets or unrelated storage."""
import argparse
import datetime
import zipfile
from pathlib import Path

from scripts.experiments.nbpo_repair_20260909.common import digest, file_hash, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("Simple package name required")
    patterns = ["configs/*.yaml", "configs/*.json", "probes/**/*.json", "evaluations/**/*.json",
        "evaluations/**/*.jsonl", "teachers/*/dataset_provenance.json", "teachers/*/inputs.json",
        "teachers/*/complete.json", "teachers/*/*/complete.json", "teachers/*/*/teacher_diagnostics.json",
        "teachers/*/*/solver/*.json", "datasets/*/precompute_manifest.json", "datasets/*/nbpo_dataset_provenance.json",
        "splits/manifest.json", "splits/benchmark_counts.json", "scores/*/manifest.json",
        "pools/*/complete.json", "pools/*/settings.json", "pools_exit.json", "scores_exit.json",
        "jobs/*/launch.json", "jobs/*/exit.json", "profiles/*/*.json", "profiles/*/runtime_rank*.jsonl",
        "arms/*/*.json", "arms/*/runtime_rank*.jsonl", "responses/*/*.manifest.json", "responses/*/settings.json",
        "provenance/inventory.json", "protocols/*.json", "protocols/*.yaml",
        "controllers/*/*.json", "evaluation/**/*.json", "evaluation/**/*.jsonl",
        "teacher_rm/*/*.json", "teacher_rm/*/summary.md", "deps_harmbench/runtime_validation.json",
        "jobs/*/environment_versions.json", "jobs/*/spec.json"]
    files = sorted({p for pattern in patterns for p in args.root.glob(pattern) if p.is_file()})
    if any(p.stat().st_size > 20_000_000 for p in files):
        raise ValueError("Unexpectedly large metadata file; inspect before packaging")
    out = args.root / "exports" / f"{args.name}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with zipfile.ZipFile(out, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            name = str(path.relative_to(args.root))
            payload = path.read_bytes()
            hashes[name] = digest(payload)
            archive.writestr(name, payload)
    write_json(out.with_suffix(".manifest.json"), {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "root": str(args.root), "archive_sha256": file_hash(out), "files": hashes,
        "note": "Live runtime files are point-in-time snapshots; final export will use completed artifacts."})
    print(f"{out}: {len(files)} files, {out.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    main()
