"""Build the per-prompt game tensors for the PROSPER-setting campaign.

PROSPER section 6.1 judges N responses from the Qwen2.5-7B-Instruct base under
one checklist item at a time (PSC). Each prompt therefore carries its OWN
objectives, and a prompt's game is played among its own responses -- there is
no separate comparator bank, so this is a self-play game and A_policy is the
antisymmetric payoff among the four responses rather than a learner-by-
comparator block.

The certified finite-pool solvers in this repository take a fixed number of
objectives, while WildChecklists prompts carry 2 to 12 items. Rather than pad
with neutral objectives -- which would hand the max-min a free zero and change
the game -- this fixes K=4 items per prompt, drawn from that prompt's own items
by a namespaced hash of the item text. On the frozen 2,000-prompt panel 1,849
prompts (92.5%) have at least four items, so almost nothing is discarded. The
chosen item texts are recorded per prompt, so the objective identities stay
prompt-specific exactly as PSC intends.

p_hat is the order-averaged estimator the campaign declares:
  p_hat(i,j) = 1/2 [ mean over i-first draws + mean over j-first draws ]
PROSPER reports a single pass. Both orders are judged here and averaged,
because this campaign's own screening showed that single-order cycle rates are
about seven times the order-averaged rate on this very dataset -- the excess is
judge position bias. Training on the single-order signal would be training on
that bias. The single-order tensors can be rebuilt from the same verdicts
without re-judging.

A pair needs every scheduled verdict parsed; a prompt is kept only when all six
pairs of all four chosen items are resolved. Nothing is imputed.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/work/sub_20260914/code")
from score_safe_pool import bt_fit, file_hash          # reuse the fitted BT projection

ROOT = Path("/work/sub_20260914")
NS_ITEM = "sub_20260914-prosper-item-choice:"
K = 4          # objectives per prompt, fixed by the solver contract
POOL = 4       # responses per prompt, PROSPER's N=4


def digest(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def load_verdicts(tags):
    """rows[prompt][item][(i,j)][order] = [values for i]"""
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    total = missing = 0
    sources = {}
    for tag in tags:
        d = ROOT / "wild_judgments" / tag
        chunks = sorted(d.glob("chunk*.jsonl"))
        if not chunks:
            raise SystemExit("no chunks under %s" % d)
        for path in chunks:
            manifest = json.loads((d / (path.stem + ".manifest.json")).read_text())
            actual = file_hash(path)
            if actual != manifest["sha256"]:
                raise SystemExit("chunk hash mismatch: %s" % path)
            sources[str(path)] = actual
            with path.open() as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    total += 1
                    if r["status"] != "ok" or r.get("value_for_i") is None:
                        missing += 1
                        continue
                    rows[r["prompt_id"]][int(r["item"])][(int(r["i"]), int(r["j"]))][
                        int(r["order"])].append(float(r["value_for_i"]))
    return rows, {"judgments_read": total, "unparsed": missing,
                  "parse_rate": (total - missing) / max(total, 1)}, sources


def p_hat(per_order):
    if 0 not in per_order or 1 not in per_order:
        return None
    return 0.5 * (float(np.mean(per_order[0])) + float(np.mean(per_order[1])))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--panel", default="wild_train2000.jsonl")
    ap.add_argument("--out", default="pros_train_v1")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--with-bt", action="store_true",
                    help="also fit the per-(prompt,item) Bradley-Terry projection. "
                         "No arm in this campaign (nbpo, fixedref, prosper) reads "
                         "r_bt, and with four responses many pairs are unanimous so "
                         "the fit runs to its 5000-iteration cap and dominates the "
                         "wall clock. Off by default; when off the r_bt key is ABSENT "
                         "from the npz rather than written as zeros, so nothing can "
                         "consume a placeholder by mistake.")
    args = ap.parse_args()

    panel_path = ROOT / "panel" / args.panel
    panel = {}
    for line in panel_path.open():
        if line.strip():
            r = json.loads(line)
            panel[r["prompt_id"]] = r

    rows, parse, sources = load_verdicts(args.tags)
    pairs_needed = list(itertools.combinations(range(POOL), 2))

    tensors, chosen_items, dropped = {}, {}, defaultdict(int)
    bt_reports = defaultdict(int)
    for pid, by_item in rows.items():
        meta = panel.get(pid)
        if meta is None:
            dropped["prompt_not_in_panel"] += 1
            continue
        items = meta["items"]
        if len(items) < K:
            dropped["fewer_than_four_items"] += 1
            continue
        order = sorted(range(len(items)),
                       key=lambda idx: (digest(NS_ITEM + items[idx]["text"]), idx))
        pick = sorted(order[:K])
        A = np.zeros((K, POOL, POOL))
        ok = True
        for k, item_index in enumerate(pick):
            per_pair = by_item.get(item_index)
            if per_pair is None:
                ok = False
                break
            for (i, j) in pairs_needed:
                entry = per_pair.get((i, j))
                p = p_hat(entry) if entry else None
                if p is None:
                    ok = False
                    break
                A[k, i, j] = p - 0.5
                A[k, j, i] = 0.5 - p
            if not ok:
                break
        if not ok:
            dropped["unresolved_pair"] += 1
            continue
        idx = np.arange(POOL)
        A[:, idx, idx] = 0.0
        if np.abs(A + np.swapaxes(A, -1, -2)).max() > 1e-12:
            raise SystemExit("payoff not antisymmetric for %s" % pid)
        if np.abs(A).max() > 0.5 + 1e-12:
            raise SystemExit("payoff outside [-0.5, 0.5] for %s" % pid)
        r_bt = None
        if args.with_bt:
            r_bt = np.zeros((K, 2, POOL))
            for k, item_index in enumerate(pick):
                obs = [(i, j, 0.5 + A[k, i, j]) for (i, j) in pairs_needed]
                # bt_fit returns (scores, report); the report is counted so a
                # non-converged projection is visible rather than silently used
                scores, report = bt_fit(obs, POOL)
                bt_reports["converged" if report["converged"] else "not_converged"] += 1
                r_bt[k, 0] = scores
                r_bt[k, 1] = scores   # self-play: one bank serves both roles
        tensors[pid] = (A, A.copy(), r_bt)
        chosen_items[pid] = [{"item_index": int(v), "text": items[v]["text"],
                              "importance": items[v].get("importance")} for v in pick]

    pids = sorted(tensors)
    if not pids:
        raise SystemExit("no prompt survived scoring")
    out = ROOT / "scores" / args.out
    out.mkdir(parents=True, exist_ok=True)
    per = (len(pids) + args.shards - 1) // args.shards
    written = []
    for s in range(args.shards):
        chunk = pids[s * per:(s + 1) * per]
        if not chunk:
            continue
        A = np.stack([tensors[p][0] for p in chunk], axis=1)
        Aref = np.stack([tensors[p][1] for p in chunk], axis=1)
        path = out / ("scores_shard%d.npz" % s)
        arrays = {"prompt_ids": np.array(chunk), "A_policy": A, "A_ref": Aref}
        shapes = {"A_policy": list(A.shape), "A_ref": list(Aref.shape)}
        if args.with_bt:
            Rbt = np.stack([tensors[p][2] for p in chunk], axis=1)
            arrays["r_bt"] = Rbt
            shapes["r_bt"] = list(Rbt.shape)
        else:
            shapes["r_bt"] = ("absent: not computed, and deliberately not written as "
                              "zeros so no arm can consume a placeholder projection")
        np.savez(path, **arrays)
        rec = {"shard": s, "prompts": len(chunk), "sha256": file_hash(path),
               "shapes": shapes,
               "gpm_teacher": {
                   "judge": "local Qwen3-14B, frozen campaign judge",
                   "estimator": ("p_hat = 1/2[mean over i-first draws + mean over "
                                 "j-first draws]; both orders judged, one draw each"),
                   "objectives": "four checklist items per prompt, prompt-specific (PSC)",
                   "single_order_variant_note": (
                       "PROSPER reports a single pass. Both orders are averaged here "
                       "because this campaign measured single-order cycle rates about 7x "
                       "the order-averaged rate on WildChecklists; the excess is position "
                       "bias. The single-order tensors are rebuildable from the same "
                       "verdicts.")},
               "bt_teacher": ({"fit": "per (prompt, item) Bradley-Terry on the six observed pairs",
                               "ridge": 1e-3, "self_play": "one response bank fills both roles"}
                              if args.with_bt else
                              {"fit": "not computed",
                               "reason": ("no arm in this campaign uses the BT projection; "
                                          "with four responses many pairs are unanimous and "
                                          "the fit runs to its iteration cap")}),
               "reference_construction": "self_play_same_four_responses",
               "item_choice_rule": ("four of the prompt's own items, ordered by "
                                    "SHA256('%s' + item text)" % NS_ITEM)}
        (out / ("complete_shard%d.json" % s)).write_text(json.dumps(rec, indent=2) + "\n")
        written.append(rec)

    report = {"out": str(out), "panel": str(panel_path),
              "panel_sha256": file_hash(panel_path),
              "tags": args.tags, "sources_sha256": sources,
              "parse": parse, "prompts_scored": len(pids),
              "prompts_dropped": dict(dropped),
              "K": K, "POOL": POOL,
              "bt_projection_convergence": dict(bt_reports),
              "shards": [{"shard": r["shard"], "prompts": r["prompts"],
                          "sha256": r["sha256"]} for r in written],
              "source_sha256": file_hash(__file__)}
    (out / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "chosen_items.json").write_text(json.dumps(chosen_items, indent=1,
                                                      ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("prompts_scored", "prompts_dropped", "parse", "K", "POOL")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
