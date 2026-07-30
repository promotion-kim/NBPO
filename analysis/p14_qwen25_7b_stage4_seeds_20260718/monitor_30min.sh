#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || exit 2
ROOT=$1; OUT=$ROOT/hourly; mkdir -p "$OUT"
while :; do
  stamp=$(date +%Y%m%dT%H%M%S%z)
  python3 - "$OUT/$stamp.json" "$ROOT" <<'PY'
import glob,json,subprocess,sys,time
out,root=sys.argv[1:]
def command(args):
    return subprocess.check_output(args,text=True).splitlines()
payload={
 "timestamp":time.time(),
 "gpus":command(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.used,memory.total","--format=csv,noheader"]),
 "processes":command(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_memory","--format=csv,noheader"]),
 "done":len(glob.glob(root+"/scheduler/*.DONE.json")),
 "failed":len(glob.glob(root+"/scheduler/*.FAILED.json")),
 "blocked":len(glob.glob(root+"/scheduler/*.BLOCKED.json")),
 "uploads":len([p for p in glob.glob(root+"/hf_uploads/s*_stage*.json") if not p.endswith("_prune.json")]),
 "pruned":len(glob.glob(root+"/hf_uploads/*_prune.json")),
 "evaluated":glob.glob(root+"/evaluation/s*/EVAL_COMPLETE"),
}
open(out,"w").write(json.dumps(payload,indent=2)+"\n")
PY
  sleep 1800
done

