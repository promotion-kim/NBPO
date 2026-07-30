#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/MNPO_rev_20260710}"
RUN_ROOT="${RUN_ROOT:-/NHNHOME/26msit001_A/BASE/AIPR/sjkim/baseline_repair_1p5b_20260714}"
PYTHON="$RUN_ROOT/venv_train/bin/python"
LOG="$RUN_ROOT/logs/eval_dispatch.log"
cd "$PROJECT_ROOT"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] waiting for the training grid queue"
while tmux has-session -t repair_grid_queue 2>/dev/null; do
  sleep 30
done

CANDIDATES=(
  repair1p5b_sppo_a_s42
  repair1p5b_sppo_b_s42
  repair1p5b_sppo_c_s42
  repair1p5b_sppo_d_s42
  repair1p5b_inpo_a_s42
  repair1p5b_inpo_b_s42
  repair1p5b_inpo_c_s42
  repair1p5b_inpo_d_s42
)
for name in "${CANDIDATES[@]}"; do
  if [[ ! -f "$RUN_ROOT/candidates/$name/model.safetensors" ]]; then
    echo "[$(date -Is)] missing completed candidate $name; evaluation cannot be complete" >&2
    exit 3
  fi
done

run_decode_wave() {
  local -a jobs=("$@")
  local -a pids=()
  local spec name gpu
  for spec in "${jobs[@]}"; do
    name="${spec%%:*}"
    gpu="${spec##*:}"
    echo "[$(date -Is)] decode $name on GPU $gpu"
    bash scripts/revision/run_baseline_repair_1p5b_decode.sh "$name" "$gpu" &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  return "$failed"
}

run_decode_wave \
  repair1p5b_sppo_a_s42:0 repair1p5b_sppo_b_s42:1 \
  repair1p5b_sppo_c_s42:2 repair1p5b_sppo_d_s42:3 || true
run_decode_wave \
  repair1p5b_inpo_a_s42:0 repair1p5b_inpo_b_s42:1 \
  repair1p5b_inpo_c_s42:2 repair1p5b_inpo_d_s42:3 || true

BASE="$RUN_ROOT/eval/generations/baseline/output_42.json"
RONPO="$RUN_ROOT/eval/generations/ronpo/output_42.json"
mkdir -p "$RUN_ROOT/eval/stability"
for core in baseline ronpo; do
  path="$RUN_ROOT/eval/generations/$core/output_42.json"
  "$PYTHON" scripts/revision/check_baseline_repair_stability.py \
    --base "$BASE" --candidate "$path" --candidate-name "$core" \
    --output "$RUN_ROOT/eval/stability/${core}_gate.json"
done

PASSING=()
for name in "${CANDIDATES[@]}"; do
  path="$RUN_ROOT/eval/generations/$name/output_42.json"
  if [[ ! -f "$path" ]]; then
    echo "[$(date -Is)] FAIL-CLOSED $name: decode output missing"
    continue
  fi
  if "$PYTHON" scripts/revision/check_baseline_repair_stability.py \
      --base "$BASE" --candidate "$path" --candidate-name "$name" \
      --output "$RUN_ROOT/eval/stability/${name}_gate.json"; then
    PASSING+=("$name")
  else
    echo "[$(date -Is)] FAIL-CLOSED $name: stability gate failed"
  fi
done

GENERATIONS=("baseline=$BASE" "ronpo=$RONPO")
for name in "${PASSING[@]}"; do
  GENERATIONS+=("$name=$RUN_ROOT/eval/generations/$name/output_42.json")
done
"$PYTHON" -m mnpo_scripts.merge_model_generations \
  --generations "${GENERATIONS[@]}" \
  --output_file "$RUN_ROOT/eval/merged_model_generations.json"

echo "[$(date -Is)] scoring model pool with three RMs; Athene uses two data-parallel shards"
bash scripts/revision/run_baseline_repair_1p5b_rm.sh skywork 0 & p_sky=$!
bash scripts/revision/run_baseline_repair_1p5b_rm.sh armo 1 & p_armo=$!
bash scripts/revision/run_baseline_repair_1p5b_rm.sh athene 2 2 0 & p_ath0=$!
bash scripts/revision/run_baseline_repair_1p5b_rm.sh athene 3 2 1 & p_ath1=$!
rm_failed=0
for pid in "$p_sky" "$p_armo" "$p_ath0" "$p_ath1"; do
  wait "$pid" || rm_failed=1
done
if [[ "$rm_failed" -ne 0 ]]; then
  echo "[$(date -Is)] at least one RM scorer failed; preserving measured shards and stopping" >&2
  exit 4
fi

"$PYTHON" - "$RUN_ROOT/eval/scored" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = []
for path in (root / "eval_athene.shard0-of-2.jsonl", root / "eval_athene.shard1-of-2.jsonl"):
    with path.open(encoding="utf-8") as stream:
        rows.extend(json.loads(line) for line in stream if line.strip())
rows.sort(key=lambda row: str(row.get("prompt_id") or row["prompt"]))
with (root / "eval_athene.jsonl").open("w", encoding="utf-8") as stream:
    for row in rows:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"merged {len(rows)} Athene prompt records")
PY

"$PYTHON" -m mnpo_scripts.evaluate_multi_objective_models \
  --scored_files \
    "skywork=$RUN_ROOT/eval/scored/eval_skywork.jsonl" \
    "athene=$RUN_ROOT/eval/scored/eval_athene.jsonl" \
    "armo=$RUN_ROOT/eval/scored/eval_armo.jsonl" \
  --output_dir "$RUN_ROOT/eval/results" \
  --baseline_model baseline \
  --bootstrap_samples 2000 \
  --bootstrap_seed 42

printf '%s\n' "${PASSING[@]}" > "$RUN_ROOT/eval/passing_candidates.txt"
"$PYTHON" scripts/revision/build_baseline_repair_1p5b_report.py \
  --run-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/final"
echo "[$(date -Is)] evaluation and bootstrap complete; passing=${PASSING[*]}"
