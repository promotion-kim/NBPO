#!/usr/bin/env bash
# Keep every GPU on THIS host busy: whenever a GPU is idle, claim the next
# unclaimed job from a shared QUEUE (atomic mkdir lock, so several hosts can
# share one queue safely) and launch it on that GPU. One launch per poll cycle
# so a freshly launched job grabs memory before the next scan. Exits when the
# queue is fully claimed.
#   QUEUE=/shared/queue.txt bash gpu_dispatcher.sh
# Queue: one shell command per line (# and blank lines ignored). Each job runs
# with env DISPATCH_GPU set to the chosen GPU index.
set -uo pipefail
QUEUE="${QUEUE:?set QUEUE}"
IDLE_MIB="${IDLE_MIB:-3000}"
POLL="${POLL:-75}"
D="$(dirname "$QUEUE")"; LOCKS="$D/locks"; mkdir -p "$LOCKS" "$D/dispatch_logs"
tagof() { printf '%s' "$1" | md5sum | cut -c1-8; }

pending() {
  grep -vE '^\s*(#|$)' "$QUEUE" | while IFS= read -r j; do
    [[ -d "$LOCKS/$(tagof "$j")" ]] || printf '%s\n' "$j"
  done
}
ngpu() { nvidia-smi --list-gpus 2>/dev/null | wc -l; }
gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '; }

echo "[dispatch] start $(hostname) $(date -Is); pending=$(pending | wc -l)"
declare -A idle
while :; do
  [[ "$(pending | wc -l)" -eq 0 ]] && { echo "[dispatch] $(hostname) queue drained $(date -Is)"; break; }
  N=$(ngpu)
  for g in $(seq 0 $((N-1))); do
    m=$(gpu_mib "$g")
    if [[ -n "$m" && "$m" -lt "$IDLE_MIB" ]]; then idle[$g]=$(( ${idle[$g]:-0} + 1 )); else idle[$g]=0; fi
  done
  for g in $(seq 0 $((N-1))); do
    [[ "${idle[$g]:-0}" -ge 2 ]] || continue          # sustained idle only (avoid train->decode dips)
    while IFS= read -r job; do
      [[ -z "$job" ]] && break
      tag=$(tagof "$job")
      if mkdir "$LOCKS/$tag" 2>/dev/null; then          # atomic claim
        echo "$(hostname) gpu$g $(date -Is)" > "$LOCKS/$tag/owner"
        echo "[dispatch] $(date +%H:%M) $(hostname) gpu$g <- $job"
        DISPATCH_GPU="$g" nohup bash -c "$job" > "$D/dispatch_logs/${tag}.log" 2>&1 &
        idle[$g]=0
        break 2                                          # one launch per cycle
      fi
    done < <(pending)
  done
  sleep "$POLL"
done
