"""The WildChecklists row of tab:dataset_readiness: wildN, wildC, wildD.

The checklist items here are prompt-specific, so this row is NOT the UF/Safe
computation run on a third dataset. The template is explicit -- "report
prompt-level variation rather than pooling checklist positions as global
parties" -- and the gamma*/TV columns stay n/a because there is no fixed
objective identity to define an aggregate finite game over. Accordingly:

  wildC  the repeated oriented 3-cycle fraction WITHIN a native item: for each
         (prompt, item) graph on the eight responses, the cycles present in
         both independent repeat panels, over the triples eligible in both.
         The unit is one item of one prompt; item k of prompt P and item k of
         prompt Q are never the same objective.
  wildD  cross-ITEM disagreement measured WITHIN a prompt: for each prompt,
         over the unordered pairs of that prompt's own items, the fraction of
         (item pair, response pair) cells where both items resolve a strict
         direction and order it oppositely. Nothing is compared across prompts.
  wildN  the prompts contributing complete graphs.

Everything else -- the order-averaged p_hat, the delta tie band, edge
resolution, the oriented-cycle enumeration and the Condorcet test -- is
imported from analyze_screen rather than reimplemented, so the three rows of
the table cannot drift apart in definition. Unparsed judgments stay missing.

wildD is reported two ways because the pooled and per-prompt weightings answer
different questions: pooling cells is the direct analogue of the Safe and UF
rows (a fraction of strictly-resolved cells), while the per-prompt mean gives
each prompt equal weight regardless of how many items it carries. The pooled
value goes in the table, and both are in the artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_screen import condorcet, edges, file_hash, oriented_cycles, p_hat


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

ROOT = Path("/work/sub_20260914")


def load(tags):
    """rows[pid][item][(i,j)][draw] = [(order, value), ...], hash-verified."""
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    total, missing, sources = 0, 0, {}
    items_seen = defaultdict(set)
    for tag in tags:
        directory = ROOT / "wild_judgments" / tag
        chunks = sorted(directory.glob("chunk*.jsonl"))
        if not chunks:
            raise SystemExit("no chunks under %s" % directory)
        for path in chunks:
            manifest_path = directory / (path.stem + ".manifest.json")
            manifest = json.loads(manifest_path.read_text())
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
                    if r["status"] != "ok" or r["value_for_i"] is None:
                        missing += 1
                        continue
                    key = (int(r["i"]), int(r["j"]))
                    items_seen[r["prompt_id"]].add(int(r["item"]))
                    rows[r["prompt_id"]][int(r["item"])][key][int(r["draw"])].append(
                        (int(r["order"]), float(r["value_for_i"])))
    parse = {"judgments_read": total, "unparsed": missing,
             "parse_rate": (total - missing) / max(total, 1)}
    return rows, parse, sources, items_seen


def duplicate_map(tag, responses):
    """prompt -> the response indices that are textually distinct, first kept."""
    path = ROOT / "responses" / tag / "responses.jsonl"
    if not path.exists():
        return {}, 0, None
    per_prompt, dup = defaultdict(dict), 0
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            e = json.loads(line)
            per_prompt[e["prompt_id"]][int(e["response_index"])] = (
                e.get("response_sha256") or digest(e["response"]))
    keep_map = {}
    for pid, idx_hash in per_prompt.items():
        first, keep = {}, []
        for idx in sorted(idx_hash):
            h = idx_hash[idx]
            if h in first:
                dup += 1
                continue
            first[h] = idx
            keep.append(idx)
        keep_map[pid] = keep
    return keep_map, dup, file_hash(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tags", nargs="+", required=True,
                    help="wild_judgments tags to merge (shards of one contract)")
    ap.add_argument("--responses-dir", default="wild200",
                    help="responses directory, read only to find duplicate texts")
    ap.add_argument("--panel-path", default="wild200.jsonl",
                    help="frozen panel, read to record item counts and confirm ids")
    ap.add_argument("--out", required=True, help="output json under analysis/")
    ap.add_argument("--delta", type=float, default=0.05,
                    help="tie band; the same declared 0.05 as the other rows")
    ap.add_argument("--responses", type=int, default=8)
    ap.add_argument("--panel-split", type=int, nargs="+", default=[0, 1],
                    help="the two independent repeat draws")
    ap.add_argument("--cell-prefix", default="wild")
    args = ap.parse_args()

    rows, parse, sources, items_seen = load(args.tags)
    panel_path = ROOT / "panel" / args.panel_path
    panel = {}
    for line in panel_path.open():
        if line.strip():
            row = json.loads(line)
            panel[row["prompt_id"]] = row
    unknown = sorted(set(rows) - set(panel))
    if unknown:
        raise SystemExit("verdicts for prompts absent from the frozen panel: %s"
                         % unknown[:5])

    responses = list(range(args.responses))
    n_pairs = len(responses) * (len(responses) - 1) // 2
    panels = [{d} for d in args.panel_split]
    dup_map, dup_count, responses_sha = duplicate_map(args.responses_dir, responses)

    eligible_single = cycles_single = 0
    eligible_repeat = cycles_repeat = 0
    eligible_repeat_unique = cycles_repeat_unique = 0
    graphs_scored = no_condorcet = strict_seen = 0
    item_graphs_complete = 0
    order_gap = []
    dis_num_pooled, dis_den_pooled = 0, 0
    per_prompt_fraction = []
    complete_prompts, cycle_items = [], 0
    per_prompt_report = {}

    for pid in sorted(rows):
        by_item = rows[pid]
        declared_items = len(panel[pid]["items"])
        edge_by_item, complete_items = {}, []
        for item in sorted(by_item):
            pairs = by_item[item]
            full = edges(pairs, args.delta)
            is_complete = (len(full) == n_pairs
                           and not any(v is None for v in full.values()))
            if is_complete:
                complete_items.append(item)
                item_graphs_complete += 1
            edge_by_item[item] = full

            for key, draws in pairs.items():
                per_order = defaultdict(list)
                for d, entries in draws.items():
                    for order, value in entries:
                        per_order[order].append(value)
                if 0 in per_order and 1 in per_order:
                    order_gap.append(abs(float(np.mean(per_order[0]))
                                         - float(np.mean(per_order[1]))))

            found_all, elig_all = oriented_cycles(full, responses)
            eligible_single += elig_all
            cycles_single += len(found_all)
            cycle_items += int(bool(found_all))
            pa, pb = (oriented_cycles(edges(pairs, args.delta, panel=p), responses)
                      for p in panels)
            eligible_repeat += min(pa[1], pb[1])
            cycles_repeat += len(pa[0] & pb[0])
            uniq = dup_map.get(pid, responses)
            if len(uniq) >= 3:
                ua, ub = (oriented_cycles(edges(pairs, args.delta, panel=p), uniq)
                          for p in panels)
                eligible_repeat_unique += min(ua[1], ub[1])
                cycles_repeat_unique += len(ua[0] & ub[0])
            weak, strict = condorcet(full, responses)
            no_condorcet += int(not weak)
            strict_seen += int(strict)
            graphs_scored += 1

        # cross-item disagreement, strictly inside this prompt
        num, den = 0, 0
        for ia, ib in combinations(sorted(edge_by_item), 2):
            ea, eb = edge_by_item[ia], edge_by_item[ib]
            for key in ea:
                a, b = ea.get(key), eb.get(key)
                if a in (None, 0) or b in (None, 0):
                    continue
                den += 1
                num += int(a != b)
        dis_num_pooled += num
        dis_den_pooled += den
        if den:
            per_prompt_fraction.append(num / den)
        if len(complete_items) == declared_items and declared_items >= 2:
            complete_prompts.append(pid)
        per_prompt_report[pid] = {
            "declared_items": declared_items,
            "items_with_verdicts": len(by_item),
            "items_complete": len(complete_items),
            "disagreement_cells": den, "disagreements": num,
            "disagreement_fraction": (num / den) if den else None}

    pooled = dis_num_pooled / max(dis_den_pooled, 1)
    result = {
        "tags": args.tags, "sources_sha256": sources,
        "panel": str(panel_path), "panel_sha256": file_hash(panel_path),
        "responses_sha256": responses_sha,
        "delta_tie_margin": args.delta,
        "unit_of_analysis": ("one checklist item of one prompt; item k of two "
                             "different prompts is never the same objective"),
        "n_prompts_read": len(rows),
        "n_prompts_complete": len(complete_prompts),
        "n_item_graphs": graphs_scored,
        "n_item_graphs_complete": item_graphs_complete,
        "items_declared_total": sum(len(panel[p]["items"]) for p in rows),
        "parse": parse,
        "C_repeated": cycles_repeat / max(eligible_repeat, 1),
        "C_single_panel": cycles_single / max(eligible_single, 1),
        "C_repeated_unique": cycles_repeat_unique / max(eligible_repeat_unique, 1),
        "eligible_triples_repeated": eligible_repeat,
        "eligible_triples_single": eligible_single,
        "eligible_triples_repeated_unique": eligible_repeat_unique,
        "cycles_repeated": cycles_repeat, "cycles_single": cycles_single,
        "duplicate_occurrences": dup_count,
        "any_cycle_item_fraction": cycle_items / max(graphs_scored, 1),
        "no_weak_condorcet_fraction": no_condorcet / max(graphs_scored, 1),
        "strict_condorcet_fraction": strict_seen / max(graphs_scored, 1),
        "graphs_scored": graphs_scored,
        "D_definition": ("within each prompt, over the unordered pairs of that "
                         "prompt's own checklist items, the fraction of "
                         "strictly-resolved (item pair, response pair) cells "
                         "ordered oppositely; pooled over prompts"),
        "D_disagreement": pooled,
        "D_disagreement_prompt_mean": (float(np.mean(per_prompt_fraction))
                                       if per_prompt_fraction else None),
        "D_disagreement_prompt_sd": (float(np.std(per_prompt_fraction, ddof=1))
                                     if len(per_prompt_fraction) > 1 else None),
        "D_prompts_contributing": len(per_prompt_fraction),
        "disagreement_denominator": dis_den_pooled,
        "mean_order_gap": float(np.mean(order_gap)) if order_gap else None,
        "gamma_and_TV": ("n/a by the template: the checklist items are "
                         "prompt-specific, so there is no fixed objective "
                         "identity to define the aggregate finite game over"),
        "per_prompt": per_prompt_report,
        "source_sha256": file_hash(__file__),
    }
    pre = args.cell_prefix
    result["cells"] = {pre + "N": len(complete_prompts),
                       pre + "C": result["C_repeated"],
                       pre + "D": result["D_disagreement"]}

    out = ROOT / "analysis" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(json.dumps({k: result[k] for k in
                      ("n_prompts_read", "n_prompts_complete", "n_item_graphs",
                       "C_repeated", "C_single_panel", "C_repeated_unique",
                       "D_disagreement", "D_disagreement_prompt_mean",
                       "disagreement_denominator", "no_weak_condorcet_fraction",
                       "mean_order_gap", "duplicate_occurrences", "parse")},
                     default=float), flush=True)
    print(json.dumps({"cells": result["cells"]}, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
