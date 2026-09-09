"""Aggregate what each arm actually did, from its own per-step runtime records."""
import json
import statistics
from pathlib import Path

ROOT = Path("/work/nbpo_repair_20260909")
ARMS = ("wbc_primary_v1", "mse_primary_v1", "wbc_short_primary_v1", "mse_short_primary_v1")


def summarize(arm, horizon):
    per_rank = {}
    for rank in range(4):
        path = ROOT / "arms" / arm / f"runtime_rank{rank}.jsonl"
        if path.exists():
            per_rank[rank] = [json.loads(line) for line in path.read_text().splitlines()]
    if not per_rank:
        return None
    rows = per_rank[0]
    # The final record is written before that step's timing and memory fields
    # exist, so every derived statistic takes only the records carrying the field.
    seconds = [r["seconds_per_update_including_instrumentation"] for r in rows
               if "seconds_per_update_including_instrumentation" in r]
    memory = [r["peak_allocated_bytes"] for r in rows if "peak_allocated_bytes" in r]
    grads = [r["preclip_grad_norm"] for r in rows if r.get("preclip_grad_norm") is not None]
    updates = [r["master_update_l2"] for r in rows if r.get("master_update_l2") is not None]
    tokens = next(r["cumulative_forward_tokens_this_rank"]["train"] for r in reversed(rows)
                  if "cumulative_forward_tokens_this_rank" in r)
    exit_path = ROOT / "jobs" / arm / "exit.json"
    exit_record = json.loads(exit_path.read_text()) if exit_path.exists() else None
    last = rows[-1]
    return {
        "arm": arm,
        "steps_recorded_per_rank": {rank: len(v) for rank, v in per_rank.items()},
        "every_rank_recorded_1_to_horizon": all(
            [r["step"] for r in v] == list(range(1, horizon + 1)) for v in per_rank.values()),
        "finished": exit_record is not None,
        "exit_code": exit_record["exit_code"] if exit_record else None,
        "wall_seconds": exit_record["seconds"] if exit_record else None,
        "gpu_hours": 4 * exit_record["seconds"] / 3600 if exit_record else None,
        "seconds_per_update": {"median": statistics.median(seconds), "mean": statistics.fmean(seconds),
                               "records": len(seconds)},
        "preclip_grad_norm": {"median": statistics.median(grads), "min": min(grads), "max": max(grads),
                              "p95": sorted(grads)[int(0.95 * len(grads))]},
        "clipped_fraction": sum(bool(r.get("clipped")) for r in rows) / len(rows),
        "master_update_l2": {"median": statistics.median(updates),
                             "sum_if_every_step_aligned": sum(updates)},
        "forward_tokens_rank0": tokens,
        "precision": {"loss": last["loss_dtype"], "masters": last["master_dtypes"],
                      "adam": last["adam_state_dtypes"],
                      "forward_parameters": last["forward_parameter_dtype"],
                      "logp": last["logp_dtypes"]},
        "rotary_buffer_restorations": last["rotary_buffer_restorations"],
        "peak_allocated_gb": max(memory) / 2**30,
    }


def main():
    out = {}
    for arm in ARMS:
        horizon = 250 if "short" in arm else 1750
        value = summarize(arm, horizon)
        if value:
            out[arm] = value
    (ROOT / "analysis_claude/runtime_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
