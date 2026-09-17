"""Generate the US/UT solver set from the verified UW one.

The UW panel's four solver files are already the verified implementations; the
only things that differ for US and UT are the objective count and the three
roots (scores, pool, restricted split). This generator makes those substitutions
with count==1 assertions on every anchor, exactly as build_pw_solvers.py does
for the per-prompt arms, so no arm can drift apart from the others in anything
except the rule it implements.

Nothing about the UW files is modified, and the generated files import their own
panel base module, so a US run cannot silently read UW scores.
"""
from __future__ import annotations

import argparse, hashlib, py_compile, subprocess
from pathlib import Path

CODE = Path("/work/sub_20260914/code")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sub(text, pairs, label):
    for old, new in pairs:
        if text.count(old) != 1:
            raise SystemExit("%s: anchor %r occurs %d times, expected 1"
                             % (label, old[:70], text.count(old)))
        text = text.replace(old, new)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", required=True, help="us1 or ut1")
    ap.add_argument("--objectives", type=int, required=True)
    ap.add_argument("--split-dir", required=True,
                    help="the restricted/certified split directory, e.g. us_v1p")
    args = ap.parse_args()

    p, K = args.panel, args.objectives
    objs = ", ".join('"item%d"' % k for k in range(K))
    written = {}

    # 1. the panel base: loaders, hashing, writers, and the objective slots
    src = CODE / "solve_pros4_targets_uw1.py"
    t = src.read_text()
    t = sub(t, [('OBJECTIVES = ("item0", "item1", "item2", "item3")',
                 "OBJECTIVES = (%s)" % objs)], src.name)
    header = ("# Generated from solve_pros4_targets_uw1.py by build_panel_solvers.py\n"
              "# -- do not edit by hand. source sha256 %s\n"
              "# change: OBJECTIVES -> item0..item%d, the %s panel's declared objective\n"
              "#         count. The rubric text behind each slot is that panel's frozen\n"
              "#         rubric, recorded in panel/%s/freeze.json; slot k of two panels\n"
              "#         is never pooled as one objective.\n"
              % (sha(src), K - 1, p.upper()[:2], args.split_dir.replace("_v1p", "_v1")))
    base_name = "solve_pros4_targets_%s.py" % p
    (CODE / base_name).write_text(header + t)
    written[base_name] = None

    # 2. the two per-prompt arms and the probe: repoint at this panel
    roots = [('("train", (ROOT / "scores/uw1", ROOT / "pools/uw1", "policy_train")),',
              '("train", (ROOT / "scores/%s", ROOT / "pools/%s", "policy_train")),' % (p, p)),
             ('("dev", (ROOT / "scores/uw1", ROOT / "pools/uw1", "policy_dev"))):',
              '("dev", (ROOT / "scores/%s", ROOT / "pools/%s", "policy_dev"))):' % (p, p)),
             ('with (ROOT / "splits/uw_v1p" / f"{split_file}.jsonl").open() as stream:',
              'with (ROOT / "splits/%s" / f"{split_file}.jsonl").open() as stream:'
              % args.split_dir),
             ("import solve_pros4_targets_uw1 as base",
              "import solve_pros4_targets_%s as base" % p),
             # a stale literal objective count silently turns into a fake
             # infeasibility on a panel with a different K
             ("K = 4", "K = len(base.OBJECTIVES)")]
    for stem in ("solve_pros4_pw_nbpo", "solve_pros4_prosper", "solve_pros4_pw_fixedref"):
        s = CODE / ("%s_uw1.py" % stem)
        if not s.exists():
            continue
        text = s.read_text()
        pairs = [r for r in roots if text.count(r[0]) == 1]
        if not any(r[0].startswith("import solve") for r in pairs):
            raise SystemExit("%s: base import anchor missing" % s.name)
        text = sub(text, pairs, s.name)
        head = ("# Generated from %s by build_panel_solvers.py -- do not edit by hand.\n"
                "# source sha256 %s\n"
                "# change: score/pool roots -> %s, restricted split -> %s, base module ->\n"
                "#         solve_pros4_targets_%s. The rule this arm implements is\n"
                "#         untouched.\n" % (s.name, sha(s), p, args.split_dir, p))
        name = "%s_%s.py" % (stem, p)
        (CODE / name).write_text(head + text)
        written[name] = None

    s = CODE / "probe_feasible_uw1.py"
    text = sub(s.read_text(),
               [("import solve_pros4_targets_uw1 as base",
                 "import solve_pros4_targets_%s as base" % p),
                ("K = 4", "K = len(base.OBJECTIVES)")], s.name)
    head = ("# Generated from probe_feasible_uw1.py by build_panel_solvers.py -- do not\n"
            "# edit by hand. source sha256 %s\n"
            "# change: base module -> solve_pros4_targets_%s. Scores, splits and output\n"
            "#         split are already flags.\n" % (sha(s), p))
    name = "probe_feasible_%s.py" % p
    (CODE / name).write_text(head + text)
    written[name] = None

    # every generated file must import and expose the right objective count
    import importlib, sys
    sys.path.insert(0, str(CODE))
    importlib.invalidate_caches()   # the generated modules did not exist at startup
    for name in written:
        py_compile.compile(str(CODE / name), doraise=True)
        written[name] = sha(CODE / name)[:16]
    mod = importlib.import_module(base_name[:-3])
    if len(mod.OBJECTIVES) != K:
        raise SystemExit("generated base declares %d objectives, wanted %d"
                         % (len(mod.OBJECTIVES), K))
    # every generated module that carries an objective count must agree with the
    # panel; a module that kept its original's literal is a fake result waiting
    # to happen, not a warning
    for name in written:
        m = importlib.import_module(name[:-3])
        for attr in ("K", "OBJECTIVES"):
            if not hasattr(m, attr):
                continue
            value = getattr(m, attr)
            got = value if isinstance(value, int) else len(value)
            if got != K:
                raise SystemExit("%s.%s is %r, which is not this panel's %d "
                                 "objectives" % (name, attr, value, K))
    for name in written:
        out = subprocess.run(["python3", str(CODE / name), "--help"],
                             capture_output=True, text=True)
        if out.returncode != 0:
            raise SystemExit("%s --help failed: %s" % (name, out.stderr[-400:]))
        if "uw1" in out.stdout:
            raise SystemExit("%s still advertises a uw1 default: %s" % (name, out.stdout))
    import json
    print(json.dumps({"panel": p, "objectives": list(mod.OBJECTIVES),
                      "generated": written}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
