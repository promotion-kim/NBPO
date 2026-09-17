"""Score the shared UF-4 pool with the frozen GPM and BT teachers.

Per prompt this produces the two tensors the finite-pool solver consumes:

  A_policy[k, x, i, j] = P_k(learner_i > comparator_j) - 1/2
  A_ref[k, x, a, b]    = P_k(comparator_a > comparator_b) - 1/2

The reference tensor is the comparator pool played against itself, so the
declared construction is ``shared_pool`` and skew symmetry is a hard requirement,
not a projection: each unordered comparator pair is judged once, written as
``+a`` and ``-a``, with an exact zero diagonal.

Both orders are evaluated for every comparison, so the antisymmetric GPM
construction is used as designed rather than assumed. BT scores one scalar per
response and caches it, because a scalar reward has no order to average over.

Nothing here fits, calibrates or selects anything: both teachers are frozen at
the checkpoints their own dev rule chose before any policy existed.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/work/uf4_20260910")
OBJECTIVES = ("instruction_following", "truthfulness", "honesty", "helpfulness")
POOL = 8


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")


def load_teacher(directory, backbone, device):
    import sys
    sys.path.insert(0, str(ROOT / "code"))
    from train_uf4_teacher import FourHeadEncoder
    report = json.loads((Path(directory) / "report.json").read_text())
    payload = torch.load(Path(directory) / "best.pt", map_location="cpu", weights_only=False)
    model = FourHeadEncoder(backbone, payload["head"])
    model.load_state_dict(payload["state_dict"])
    model.eval().to(device)
    return model, {"dir": str(directory), "head": payload["head"], "step": payload["step"],
                   "checkpoint_sha256": file_hash(Path(directory) / "best.pt"),
                   "report_sha256": file_hash(Path(directory) / "report.json"),
                   "selected_mean_nll": report["selected"]["_mean_nll"],
                   "selected_worst_nll": report["selected"]["_worst_nll"]}


class Serializer:
    """Same markers, same budget rule and same order as the teacher saw in training."""

    def __init__(self, tokenizer, max_length, instruction_cap):
        import sys
        sys.path.insert(0, str(ROOT / "code"))
        from train_uf4_teacher import Collator
        self.gpm = Collator(tokenizer, "gpm", max_length, instruction_cap)
        self.bt = Collator(tokenizer, "bt", max_length, instruction_cap)
        self.tok = tokenizer

    def pair_rows(self, instruction, left, right):
        return {"instruction": instruction, "response_a": left, "response_b": right,
                "target": {o: 0.5 for o in OBJECTIVES}, "mask": {o: True for o in OBJECTIVES}}


@torch.no_grad()
def run_batches(model, collator, rows, device, micro, audit):
    """Return (n_rows, 4) logits, in the order the rows were given."""
    import sys
    sys.path.insert(0, str(ROOT / "code"))
    from train_uf4_teacher import autocast, merge_audit
    out = []
    for lo in range(0, len(rows), micro):
        batch = collator(rows[lo:lo + micro])
        merge_audit(audit, batch)
        with autocast(device):
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        out.append(logits.float().cpu())
    return torch.cat(out) if out else torch.zeros((0, len(OBJECTIVES)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="v1")
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--out-name", default="v1")
    ap.add_argument("--gpm-dir", default=str(ROOT / "teacher/gpm_s42"))
    ap.add_argument("--bt-dir", default=str(ROOT / "teacher/bt_s42"))
    ap.add_argument("--backbone", default=str(ROOT / "assets/ModernBERT-base"))
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--instruction-cap", type=int, default=2048)
    ap.add_argument("--micro-pairs", type=int, default=8)
    ap.add_argument("--max-chunks", type=int)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, local_files_only=True)
    serializer = Serializer(tokenizer, args.max_length, args.instruction_cap)
    gpm, gpm_info = load_teacher(args.gpm_dir, args.backbone, device)
    bt, bt_info = load_teacher(args.bt_dir, args.backbone, device)
    if gpm_info["head"] != "gpm" or bt_info["head"] != "bt":
        raise ValueError("Teacher directories do not carry the heads they claim")

    pool_dir = ROOT / "pools" / args.pool / f"shard{args.shard}"
    out = ROOT / "scores" / args.out_name / f"shard{args.shard}"
    out.mkdir(parents=True, exist_ok=True)
    chunks = sorted(pool_dir.glob("chunk*.jsonl"))
    if args.max_chunks:
        chunks = chunks[:args.max_chunks]
    if not chunks:
        raise ValueError(f"No pool chunks under {pool_dir}")

    cross_pairs = [(i, j) for i in range(POOL) for j in range(POOL)]
    self_pairs = list(itertools.combinations(range(POOL), 2))
    audit = {}
    start = time.monotonic()
    totals = {"prompts": 0, "gpm_sequences": 0, "bt_sequences": 0, "seconds": 0.0}
    for chunk_path in chunks:
        stem = chunk_path.stem
        destination = out / f"{stem}.npz"
        metadata = out / f"{stem}.manifest.json"
        if destination.exists() and metadata.exists():
            if file_hash(destination) == json.loads(metadata.read_text())["sha256"]:
                continue
            raise ValueError(f"Partial or corrupt score chunk {destination}")
        events = {}
        with chunk_path.open() as stream:
            for line in stream:
                event = json.loads(line)
                events.setdefault(event["prompt_id"], {}).setdefault(event["role"], {})[
                    event["sample_index"]] = event
        prompt_ids = sorted(events)
        for pid in prompt_ids:
            roles = events[pid]
            if sorted(roles) != ["comparator", "learner"] or any(
                    sorted(roles[r]) != list(range(POOL)) for r in roles):
                raise ValueError(f"Incomplete pool for prompt {pid}")

        before = time.monotonic()
        K, X = len(OBJECTIVES), len(prompt_ids)
        A_policy = np.zeros((K, X, POOL, POOL), dtype=np.float64)
        A_ref = np.zeros((K, X, POOL, POOL), dtype=np.float64)
        r_bt = np.zeros((K, X, 2, POOL), dtype=np.float64)
        n_gpm = n_bt = 0
        for x, pid in enumerate(prompt_ids):
            learner = [events[pid]["learner"][i]["response"] for i in range(POOL)]
            comparator = [events[pid]["comparator"][j]["response"] for j in range(POOL)]
            instruction = events[pid]["learner"][0]["prompt"]

            rows = [serializer.pair_rows(instruction, learner[i], comparator[j])
                    for i, j in cross_pairs]
            rows += [serializer.pair_rows(instruction, comparator[a], comparator[b])
                     for a, b in self_pairs]
            logits = run_batches(gpm, serializer.gpm, rows, device, args.micro_pairs, audit)
            n_gpm += 2 * len(rows)
            probability = torch.sigmoid(logits).numpy()
            for index, (i, j) in enumerate(cross_pairs):
                A_policy[:, x, i, j] = probability[index] - 0.5
            for offset, (a, b) in enumerate(self_pairs):
                value = probability[len(cross_pairs) + offset] - 0.5
                A_ref[:, x, a, b] = value
                A_ref[:, x, b, a] = -value          # one judged pair, written both ways

            # BT: one scalar per response. The pair collator emits (A, B) per row,
            # so 8 rows recover all 16 responses with no wasted forward.
            bt_rows = [serializer.pair_rows(instruction, learner[i], comparator[i])
                       for i in range(POOL)]
            bt_logits_pairs = []
            for lo in range(0, len(bt_rows), args.micro_pairs):
                batch = serializer.bt(bt_rows[lo:lo + args.micro_pairs])
                from train_uf4_teacher import autocast, merge_audit
                merge_audit(audit, batch)
                with torch.no_grad(), autocast(device):
                    scores = bt.encoder(input_ids=batch["input_ids"].to(device),
                                        attention_mask=batch["attention_mask"].to(device))
                    pooled = bt.score(scores.last_hidden_state[:, 0]).float().cpu()
                bt_logits_pairs.append(pooled)
            pooled = torch.cat(bt_logits_pairs).numpy()     # (2*POOL, K), interleaved A,B
            n_bt += pooled.shape[0]
            r_bt[:, x, 0, :] = pooled[0::2].T                # learners
            r_bt[:, x, 1, :] = pooled[1::2].T                # comparators

        skew = float(np.abs(A_ref + np.swapaxes(A_ref, -1, -2)).max())
        diagonal = float(np.abs(np.einsum("kxii->kxi", A_ref)).max())
        if skew > 0 or diagonal > 0:
            raise ValueError(f"Reference tensor is not exactly skew symmetric: {skew}, {diagonal}")
        np.savez_compressed(destination, A_policy=A_policy, A_ref=A_ref, r_bt=r_bt,
                            prompt_ids=np.array(prompt_ids, dtype=object), allow_pickle=True)
        elapsed = time.monotonic() - before
        report = {"sha256": file_hash(destination), "n_prompts": X,
                  "gpm_sequences": n_gpm, "bt_sequences": n_bt, "seconds": elapsed,
                  "gpm_sequences_per_second": n_gpm / elapsed,
                  "pool_chunk_sha256": file_hash(chunk_path),
                  "A_policy_abs_max": float(np.abs(A_policy).max()),
                  "A_ref_skew_residual": skew, "A_ref_diagonal_residual": diagonal,
                  "bt_score_std": float(r_bt.std())}
        write_json(metadata, report)
        totals["prompts"] += X
        totals["gpm_sequences"] += n_gpm
        totals["bt_sequences"] += n_bt
        totals["seconds"] += elapsed
        print(json.dumps({"chunk": stem, "prompts": X, "gpm_seq": n_gpm, "bt_seq": n_bt,
                          "seconds": round(elapsed, 1),
                          "gpm_seq_per_s": round(n_gpm / elapsed, 1)}), flush=True)

    summary = {"shard": args.shard, "pool_dir": str(pool_dir), "objectives": list(OBJECTIVES),
               "reference_construction": "shared_pool",
               "gpm_teacher": gpm_info, "bt_teacher": bt_info,
               "totals": totals, "length_budget": audit,
               "wall_seconds": time.monotonic() - start,
               "source_sha256": file_hash(__file__)}
    write_json(out / f"complete_shard{args.shard}.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "gpm_teacher"}), flush=True)


if __name__ == "__main__":
    main()
