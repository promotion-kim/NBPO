#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 CHECKPOINT_DIR [CHECKPOINT_DIR ...]" >&2
  exit 2
fi

KUBECONFIG_PATH="${KUBECONFIG_PATH:-mlxp/aipr-kubeconfig.yaml}"
KUBE_NS="${KUBE_NS:-p-aipr3}"
KUBE_POD="${KUBE_POD:-mnpo-rev-q3-sppo-avg-s42-2gpu-mjzpp}"
KUBE_CONTAINER="${KUBE_CONTAINER:-main}"
SRC_ROOT="${SRC_ROOT:-/data/mnpo/revision_qwen3_8b/full_iter1/train}"
DST_ROOT="${DST_ROOT:-/NHNHOME/WORKSPACE/26msit001_A/mnpo/revision_qwen3_8b/full_iter1/train}"
B200_HOST="${B200_HOST:-59.150.33.1}"
B200_PORT="${B200_PORT:-50104}"
B200_USER="${B200_USER:-aipr_lab}"
B200_KEY="${B200_KEY:-nhn/sjkim-interactive_key}"
HAVOK_HOST="${HAVOK_HOST:-59.29.246.23}"
HAVOK_PORT="${HAVOK_PORT:-3000}"
HAVOK_USER="${HAVOK_USER:-sjkim}"
HAVOK_CM="${HAVOK_CM:-/tmp/mnpo_havok_cm}"
CHUNK_BYTES="${CHUNK_BYTES:-16777216}"
DD_BS="${DD_BS:-16777216}"

ssh_b200() {
  ssh -F /dev/null \
    -i "$B200_KEY" \
    -o UserKnownHostsFile=/dev/null \
    -o StrictHostKeyChecking=no \
    -o LogLevel=ERROR \
    -o ConnectTimeout=30 \
    -o ProxyCommand="ssh -F /dev/null -S $HAVOK_CM -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o LogLevel=ERROR -W %h:%p $HAVOK_USER@$HAVOK_HOST -p $HAVOK_PORT" \
    -p "$B200_PORT" "$B200_USER@$B200_HOST" "$@"
}

kexec() {
  kubectl --kubeconfig "$KUBECONFIG_PATH" -n "$KUBE_NS" exec "$KUBE_POD" -c "$KUBE_CONTAINER" -- "$@"
}

copy_file() {
  local rel_dir="$1"
  local file="$2"
  local src_path="$SRC_ROOT/$rel_dir/$file"
  local dst_dir="$DST_ROOT/$rel_dir"
  local dst_path="$dst_dir/$file"
  local tmp_path="$dst_path.tmp"
  local size
  size="$(kexec stat -c%s "$src_path")"

  local remote_size
  remote_size="$(ssh_b200 "if [ -f '$dst_path' ]; then stat -c%s '$dst_path'; else echo -1; fi")"
  if [[ "$remote_size" == "$size" ]]; then
    echo "[skip] $rel_dir/$file size=$size"
    return
  fi

  echo "[copy] $rel_dir/$file size=$size"
  ssh_b200 "mkdir -p '$dst_dir'"

  local tmp_existing
  tmp_existing="$(ssh_b200 "if [ -f '$tmp_path' ]; then stat -c%s '$tmp_path'; else echo 0; fi")"
  local offset=0
  if (( tmp_existing > 0 && tmp_existing < size )); then
    offset=$(( (tmp_existing / CHUNK_BYTES) * CHUNK_BYTES ))
    echo "  [resume] tmp_size=$tmp_existing resume_offset=$offset"
    ssh_b200 "truncate -s '$offset' '$tmp_path'"
  else
    ssh_b200 ": > '$tmp_path'"
  fi

  local idx=0
  while (( offset < size )); do
    local remain=$(( size - offset ))
    local chunk=$CHUNK_BYTES
    if (( remain < chunk )); then
      chunk=$remain
    fi
    local skip_blocks=$(( offset / DD_BS ))
    local count_blocks=$(( (chunk + DD_BS - 1) / DD_BS ))
    echo "  [chunk] $idx offset=$offset bytes=$chunk"
    kexec dd "if=$src_path" "bs=$DD_BS" "skip=$skip_blocks" "count=$count_blocks" status=none \
      | ssh_b200 "cat >> '$tmp_path'"
    offset=$(( offset + chunk ))
    idx=$(( idx + 1 ))
  done

  local tmp_size
  tmp_size="$(ssh_b200 "stat -c%s '$tmp_path'")"
  if [[ "$tmp_size" != "$size" ]]; then
    echo "[fatal] size mismatch for $rel_dir/$file: tmp=$tmp_size expected=$size" >&2
    exit 1
  fi
  ssh_b200 "mv '$tmp_path' '$dst_path'"
  echo "[done] $rel_dir/$file"
}

for rel_dir in "$@"; do
  echo "==== $rel_dir ===="
  mapfile -t files < <(kexec bash -lc "cd '$SRC_ROOT/$rel_dir' && find . -maxdepth 1 -type f -printf '%f\n' | sort")
  for file in "${files[@]}"; do
    copy_file "$rel_dir" "$file"
  done
done
