"""Build all three per-prompt solvers from the pristine source in one pass.

Successive small patches on generated copies is what produced the last two
failures: a guard left between two anchors broke the weight_l1 block, and a
heuristic region search clobbered _init. Every change is applied here from
/work/uf4_20260910/code/solve_prosper_targets.py, each anchored on an exact
string that must appear exactly once, so a drifted source raises instead of
producing a subtly wrong file.

Shared changes: base module, this campaign's score/pool/split directories, a
--shards flag replacing the hardcoded 4, and an optional matched multiplier norm.
Per arm: the aggregation rule, the representation, and every recorded label.
"""
from pathlib import Path
import ast, hashlib

import argparse

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--score-root", default="scores/pros_scores_all")
_ap.add_argument("--pool-root", default="pools/pros2_pool_v1")
_ap.add_argument("--splits", default="splits/pros_v1f")
_ap.add_argument("--shards", type=int, default=8)
_ap.add_argument("--suffix", default="",
                 help="appended to each generated file name, e.g. _v2")
_args = _ap.parse_args()

SRC = Path("/work/uf4_20260910/code/solve_prosper_targets.py")
C = Path("/work/sub_20260914/code")
src = SRC.read_text()
sha = hashlib.sha256(SRC.read_bytes()).hexdigest()

SHARED = [
 ("import solve_uf4_targets as base",
  "import solve_pros4_targets%s as base" % _args.suffix),
 ('default=ROOT / "targets/nash_v1/complete.json")', "default=None)"),
 ('("train", (ROOT / "scores/v1", ROOT / "pools/v1", "policy_train")),',
  '("train", (ROOT / "%s", ROOT / "%s", "policy_train")),' % (_args.score_root, _args.pool_root)),
 ('("dev", (ROOT / "scores/dev_v1", ROOT / "pools/dev_v1", "policy_dev"))',
  '("dev", (ROOT / "%s", ROOT / "%s", "policy_dev"))' % (_args.score_root, _args.pool_root)),
 ('ROOT / "splits/v1"', 'ROOT / "%s"' % _args.splits),
 ("base.load_scores(str(score_root), 4)", "base.load_scores(str(score_root), args.shards)"),
 ("base.load_pool(str(pool_root), 4)", "base.load_pool(str(pool_root), args.shards)"),
 ("                for row in rows:\n"
  "                    a, b = row[\"chosen_candidate_index\"], row[\"rejected_candidate_index\"]",
  "                for row in rows:\n"
  "                    # round masses to the loader's ten decimals and rebuild the\n"
  "                    # target from the rounded values; a per-prompt Nash solve\n"
  "                    # concentrates mass and the identity otherwise fails after the parse\n"
  "                    row = base.quantize_canonical_row(row)\n"
  "                    a, b = row[\"chosen_candidate_index\"], row[\"rejected_candidate_index\"]"),
 ('    ap.add_argument("--out-name", default="prosper_v1")',
  '    ap.add_argument("--out-name", default="prosper_v1")\n'
  '    ap.add_argument("--shards", type=int, default=8,\n'
  '                    help="score and pool shard count; 8 once the 400-prompt "\n'
  '                         "extension is merged in as shards 4..7")\n'
  '    ap.add_argument("--weight-l1", type=float, default=None,\n'
  '                    help="matched multiplier norm as a literal. Only absolute_maxmin "\n'
  '                         "uses it; nash derives its own dual.")'),
]

L1_REGION = '''    source = json.loads(args.weight_l1_from.read_text())
    if source.get("aggregation") != "nash":
        raise ValueError("--weight-l1-from must point at a Nash solution")
    weight_l1 = float(source["splits"]["train"]["certificate"]["lambda_l1"])
    print(json.dumps({"matched_weight_l1": weight_l1, "from": str(args.weight_l1_from)}), flush=True)
'''
L1_NASH = '''    # nash derives its own dual: no multiplier norm is read, and none is used.
    # The global NBPO arm this was matched to is uncertified (independent
    # stationarity 13.224), and nothing here is matched to an uncertified value.
    weight_l1 = None
    print(json.dumps({"weight_l1": None,
                      "note": "unused: nash derives its own dual"}), flush=True)
'''
L1_MAXMIN = '''    if args.weight_l1 is not None:
        weight_l1 = float(args.weight_l1)
        print(json.dumps({"matched_weight_l1": weight_l1,
                          "from": "literal --weight-l1"}), flush=True)
    else:
        if args.weight_l1_from is None:
            raise SystemExit("absolute_maxmin needs --weight-l1 or --weight-l1-from")
        source = json.loads(args.weight_l1_from.read_text())
        if source.get("aggregation") != "nash":
            raise ValueError("--weight-l1-from must point at a Nash solution")
        weight_l1 = float(source["splits"]["train"]["certificate"]["lambda_l1"])
        print(json.dumps({"matched_weight_l1": weight_l1,
                          "from": str(args.weight_l1_from)}), flush=True)
'''

AGG_CALL = '''        rep, "absolute_maxmin", eta=_SHARED["eta"], inner_solver="exact",
        dual_solver="root", dual_tol=1e-10, M=_SHARED["M"], inner_workers=1,
        probability_floor=_SHARED["floor"], weight_l1=_SHARED["weight_l1"], log_every=0)'''
AGG_NASH = '''        rep, "nash", eta=_SHARED["eta"], inner_solver="exact",
        dual_solver="root", dual_tol=1e-10, M=_SHARED["M"], inner_workers=1,
        probability_floor=_SHARED["floor"], log_every=0)'''
REP_ADAPT = '''    rep = AdaptiveGameRepresentation(
        torch.from_numpy(np.ascontiguousarray(A)), torch.from_numpy(np.ascontiguousarray(Aref)),
        uniform_policy(1, POOL),
        torch.full((len(OBJECTIVES),), _SHARED["beta"], dtype=torch.float64),
        reference_construction="shared_pool")'''
REP_FIXED = '''    rep = FixedReferenceRepresentation(
        torch.from_numpy(np.ascontiguousarray(A)), torch.from_numpy(np.ascontiguousarray(Aref)),
        uniform_policy(1, POOL),
        reference_construction="shared_pool")'''
IMP_OLD = "from mnpo_scripts.nbpo_representations import AdaptiveGameRepresentation"
IMP_NEW = ("from mnpo_scripts.nbpo_representations import (AdaptiveGameRepresentation,\n"
           "                                               FixedReferenceRepresentation)")

for anchor, _ in SHARED:
    if src.count(anchor) != 1:
        raise SystemExit("shared anchor %r appears %d times" % (anchor[:50], src.count(anchor)))
for anchor, label in ((L1_REGION, "l1 region"), (AGG_CALL, "aggregation call"),
                      (REP_ADAPT, "representation"), (IMP_OLD, "import")):
    if src.count(anchor) != 1:
        raise SystemExit("%s appears %d times" % (label, src.count(anchor)))
if src.count('"prompt_wise_absolute_maxmin"') != 4:
    raise SystemExit("expected four aggregation labels, found %d"
                     % src.count('"prompt_wise_absolute_maxmin"'))


# The source is PROSPER's solver, so its faithfulness note describes max-min. The
# generator rewrites the aggregation labels, and until now it left that sentence
# alone, which shipped a false description of the method inside the two Nash arms.
NOTE_SRC = (
    '        "faithfulness_note": ("PROSPER\'s inner minimisation over w in Delta_K is the per-prompt "\n'
    '                              "worst objective, so this is absolute_maxmin solved per prompt "\n'
    '                              "with no shared dual. It is NOT the global game-maxmin control."),\n')
FAITHFULNESS = {
    "pw_nbpo": (
        '        "faithfulness_note": ("Nash aggregation over the same objective-wise game "\n'
        '                              "values, with the dual multipliers fitted per prompt and "\n'
        '                              "no vector shared across prompts. It is NOT PROSPER\'s "\n'
        '                              "max-min rule and NOT the global shared-weight control."),\n'),
    "pw_fixedref": (
        '        "faithfulness_note": ("Nash aggregation per prompt against a frozen reference "\n'
        '                              "policy rather than an adaptive one. It is NOT PROSPER\'s "\n'
        '                              "max-min rule and NOT the adaptive-reference arm."),\n'),
}


def build(kind):
    t = src
    for old, new in SHARED:
        t = t.replace(old, new, 1)
    if kind == "prosper":
        t = t.replace(L1_REGION, L1_MAXMIN, 1)
        label, rep_label, change = "prompt_wise_absolute_maxmin", "adaptive_game", "MaxEntBW"
    else:
        t = t.replace(L1_REGION, L1_NASH, 1)
        t = t.replace(AGG_CALL, AGG_NASH, 1)
        if kind == "pw_fixedref":
            t = t.replace(REP_ADAPT, REP_FIXED, 1).replace(IMP_OLD, IMP_NEW, 1)
            label, rep_label, change = ("prompt_wise_nash_fixed_reference",
                                        "fixed_reference", "per-prompt Nash, fixed reference")
        else:
            label, rep_label, change = ("prompt_wise_nash", "adaptive_game",
                                        "per-prompt Nash, adaptive game")
        t = t.replace('"prompt_wise_absolute_maxmin"', '"%s"' % label)
        t = t.replace('"representation": "adaptive_game"', '"representation": "%s"' % rep_label)
        t = t.replace(NOTE_SRC, FAITHFULNESS[kind], 1)
    head = ("# Generated from solve_prosper_targets.py by build_pw_solvers.py -- do not edit.\n"
            "# source sha256 %s\n"
            "# arm: %s\n"
            "# shared changes: base module -> solve_pros4_targets; score/pool/split roots ->\n"
            "#   pros_scores_all / pros2_pool_v1 / splits/pros_v1f; hardcoded 4 shards ->\n"
            "#   --shards (default 8); matched multiplier norm optional.\n"
            "# Objectives are item0..item3, four of each prompt's OWN checklist items\n"
            "# (PROSPER PSC); item k of two prompts is never pooled as one objective.\n"
            % (sha, change))
    return head + t


for kind, name in (("prosper", "solve_pros4_prosper.py"),
                   ("pw_nbpo", "solve_pros4_pw_nbpo.py"),
                   ("pw_fixedref", "solve_pros4_pw_fixedref.py")):
    text = build(kind)
    ast.parse(text)
    body = text[text.index("def main"):]
    if kind == "prosper":
        assert 'rep, "absolute_maxmin"' in text and 'rep, "nash"' not in text
        assert "args.weight_l1 is not None" in body
        assert '"prompt_wise_absolute_maxmin"' in text
    else:
        assert 'rep, "nash"' in text and '"absolute_maxmin"' not in text
        assert "weight_l1 = None" in body and "source = json.loads" not in body
        assert '"prompt_wise_absolute_maxmin"' not in text
    if kind == "pw_fixedref":
        assert "FixedReferenceRepresentation(\n" in text
        assert "AdaptiveGameRepresentation(\n" not in text
    assert "args.shards" in text and "load_scores(str(score_root), 4)" not in text
    assert _args.score_root in text, 'score root %s missing' % _args.score_root
    assert _args.splits in text, 'splits %s missing' % _args.splits
    stem, ext = name.rsplit(".", 1)
    (C / ("%s%s.%s" % (stem, _args.suffix, ext))).write_text(text)
    print("  %-30s %s  ok" % (name.replace(".py", _args.suffix + ".py"),
                              hashlib.sha256(text.encode()).hexdigest()[:16]))
print("  pristine source %s" % sha[:16])
