#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/work/campaign_20260825/neural_bridge}
BT=$ROOT/selector_fp32
MSE=$ROOT/selector_mse_fallback
CYCLE=$ROOT/cycle_diagnostic
CROSS=$ROOT/crossjudge_llama70
RACO=/work/campaign_20260824

selector_terminal() {
  local root=$1
  [ -f "$root/all_complete" ] || [ -f "$root/stopped_fit_gate_fail" ] || \
    [ -f "$root/stopped_exact_pool_gate_fail" ] || \
    [ -f "$root/runtime_failed" ] || [ -f "$root/fit_runtime_failed" ]
}

while :; do
  if [ -f "$BT/paper_gate_pass" ]; then SELECTOR=$BT; break; fi
  if [ -f "$MSE/paper_gate_pass" ]; then SELECTOR=$MSE; break; fi
  if selector_terminal "$BT" && \
     { [ -f "$MSE/stopped_bt_phase0_passed" ] || selector_terminal "$MSE" || \
       [ -f "$MSE/stopped_no_valid_mse_phase0" ]; }; then
    mkdir -p "$RACO"; date -Is > "$RACO/raco_frontier_stopped_no_valid_selector"
    exit 0
  fi
  sleep 30
done

# Give the already-preregistered cyclic diagnostic priority on the four H200s.
while [ ! -f "$CYCLE/all_complete" ] && [ ! -f "$CYCLE/stopped_phase0_gate_fail" ] && \
      [ ! -f "$CYCLE/stopped_no_valid_phase0" ]; do sleep 30; done
while [ ! -f "$CROSS/all_complete" ] && [ ! -f "$CROSS/stopped_no_valid_selector" ]; do sleep 30; done
if [ ! -f "$CROSS/paper_gate_pass" ]; then
  mkdir -p "$RACO"; date -Is > "$RACO/raco_frontier_stopped_crossjudge_gate_fail"
  exit 0
fi

printf '%s  %s\n' \
  15a0ea453b2eebd1e811cab400a6ccb4bd04c46b4cce4f44eb67fa625dd8b9ed \
  "$ROOT/raco_cpu_smoke.json" | sha256sum -c -
python3 - "$ROOT/raco_cpu_smoke.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
raise SystemExit(0 if x.get('all_pass') else 'canonical RACO CPU smoke did not pass')
PY

mkdir -p "$RACO/logs"
cp "$ROOT/RACO_PREREG.md" "$RACO/RACO_PREREG.md"
cp "$ROOT/raco_run_lock.json" "$RACO/raco_run_lock.json"
cp "$ROOT/RACO_SHA256SUMS.pre_train" "$RACO/RACO_SHA256SUMS.pre_train"
cp "$ROOT/launch_raco_frontier_mlxp.sh" "$RACO/launch_raco_frontier_mlxp.sh"
(cd "$RACO" && printf '%s  %s\n' \
  33cc5c617a0d315594823ada0f3fbdb388e22c4b1267f85428dd130c63969806 RACO_PREREG.md \
  513dc43d8811f3326c1ebbb9d13d18dfe27cdc1b7f59cf07543ae97def927da5 raco_run_lock.json \
  58a73da984af63945e9225eb5bba0fa6f48370c1b9e314cc75b713d0a7b050f5 launch_raco_frontier_mlxp.sh | \
  sha256sum -c -) > "$RACO/logs/raco_preflight_hash.log"
python3 - "$SELECTOR/gate.json" "$RACO/raco_authorization.json" <<'PY'
import hashlib,json,sys
p=sys.argv[1]
with open(p,'rb') as f: digest=hashlib.sha256(f.read()).hexdigest()
gate=json.load(open(p))
if not gate.get('paper_gate_pass',False): raise SystemExit('selector gate changed before launch')
json.dump({'selector_gate':p,'selector_gate_sha256':digest,
           'authorized':True,'selection_role':'validity_only'},open(sys.argv[2],'w'),indent=2)
open(sys.argv[2],'a').write('\n')
PY
bash "$RACO/launch_raco_frontier_mlxp.sh"
