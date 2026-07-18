#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 ROOT HOST" >&2; exit 2; }
ROOT=$1; HOST=$2; OUT=$ROOT/stage4/hourly; mkdir -p "$OUT"
while :; do
  stamp=$(date +%Y%m%dT%H%M%S%z)
  python3 - "$OUT/${stamp}_${HOST}.json" "$ROOT" "$HOST" <<'PY'
import glob,json,subprocess,sys,time
out,root,host=sys.argv[1:]
def cmd(x): return subprocess.check_output(x,text=True).splitlines()
d={"timestamp":time.time(),"host":host,
   "gpus":cmd(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.used,memory.total","--format=csv,noheader"]),
   "processes":cmd(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_memory","--format=csv,noheader"]),
   "done":sorted(glob.glob(root+"/stage4/scheduler/*.DONE.json")),
   "failed":sorted(glob.glob(root+"/stage4/scheduler/*.FAILED.json")),
   "uploads":sorted(glob.glob(root+"/stage4/hf_uploads/stage[1-4]_*.json"))}
open(out,"w").write(json.dumps(d,indent=2)+"\n")
PY
  sleep 1800
done
