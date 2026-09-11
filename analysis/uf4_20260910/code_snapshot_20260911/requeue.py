"""Requeue a job whose script was fixed.

The controller skips a FAILED job forever unless the spec bytes change -- that
is deliberate, so a broken job does not spin. Recording why the retry exists in
the spec both changes the bytes and leaves the reason in the artifact.
"""
import json, pathlib, sys

Q = pathlib.Path("/work/uf4_20260910/jobs/queue")
job_id, reason = sys.argv[1], sys.argv[2]
for path in sorted(Q.glob("*.json")):
    spec = json.loads(path.read_text())
    if spec["job_id"] != job_id:
        continue
    out = pathlib.Path(spec["artifacts"][0]).parent
    if out.exists():
        sys.exit(f"{out} exists; the script refuses to overwrite it. Inspect before retrying.")
    spec["retry_reason"] = reason
    spec["retry"] = int(spec.get("retry", 0)) + 1
    path.write_text(json.dumps(spec, indent=2) + "\n")
    print(json.dumps({"requeued": job_id, "retry": spec["retry"], "spec": path.name}))
    break
else:
    sys.exit(f"no spec for {job_id}")
