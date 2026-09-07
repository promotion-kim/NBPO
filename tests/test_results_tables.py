"""Generated result tables: an incomplete cell must never become a number.

The failure this guards against is the one that is easiest to commit by
accident and hardest to notice later -- a table cell filled from two seeds, or
from the best seed, or from an artifact left behind by a run that crashed. The
generator is built so that cannot happen, and these tests hold it to that.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
MOD = "scripts.experiments.iclr2027_table1_v2.build_results_tables"
SEEDS = (42, 43, 44)


def build(root: Path, out: Path):
    r = subprocess.run([sys.executable, "-m", MOD, "--root", str(root),
                        "--out-dir", str(out)],
                       cwd=REPO, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads((out / "table_status.json").read_text())


def write_eval(root: Path, method: str, seed: int, worst: float):
    d = root / "eval" / f"{method}_s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "objective_win_rates.json").write_text(json.dumps({
        "worst_objective": worst, "average_objective": worst + 0.02,
        "objectives": {"instruction_following": worst + 0.03, "truthfulness": worst,
                       "honesty": worst + 0.01, "helpfulness": worst + 0.04}}))


def registry(root: Path, entries):
    (root / "run_registry.jsonl").write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n")


def test_empty_root_yields_all_pending(tmp_path):
    st = build(tmp_path, tmp_path / "out")
    assert st["rows_complete"] == 0
    assert st["manuscript_update_allowed"] is False
    tex = (tmp_path / "out" / "table1_generated.tex").read_text()
    body = [l for l in tex.splitlines() if not l.startswith("%")]
    assert "\n".join(body).count("\\pending{}") == 20   # 10 rows x 2 columns
    assert not any(any(ch.isdigit() for ch in l) for l in body if l.endswith("\\\\"))


def test_two_of_three_seeds_is_still_pending(tmp_path):
    """The specific mistake this file exists for."""
    for seed in (42, 43):
        write_eval(tmp_path, "nbpo", seed, 0.55)
    st = build(tmp_path, tmp_path / "out")
    assert st["per_row"]["nbpo"]["n_seeds"] == 2
    assert st["per_row"]["nbpo"]["complete"] is False
    assert st["per_row"]["nbpo"]["missing_seeds"] == [44]
    tex = (tmp_path / "out" / "table1_generated.tex").read_text()
    nbpo_line = [l for l in tex.splitlines() if l.startswith("NBPO ")][0]
    assert "\\pending{}" in nbpo_line and "0.5" not in nbpo_line
    # ... and the CSV leaves it empty rather than averaging two seeds
    rows = (tmp_path / "out" / "table1_mean_std.csv").read_text().splitlines()
    nbpo = [r for r in rows if r.startswith("nbpo,")][0]
    assert ",False," in nbpo
    assert nbpo.rstrip().endswith(",,,,")


def test_three_seeds_completes_the_row(tmp_path):
    for seed, w in zip(SEEDS, (0.54, 0.56, 0.55)):
        write_eval(tmp_path, "nbpo", seed, w)
    st = build(tmp_path, tmp_path / "out")
    assert st["per_row"]["nbpo"]["complete"] is True
    tex = (tmp_path / "out" / "table1_generated.tex").read_text()
    nbpo_line = [l for l in tex.splitlines() if l.startswith("NBPO ")][0]
    assert "\\pending{}" not in nbpo_line
    assert "0.550" in nbpo_line and "$\\pm$" in nbpo_line


def test_an_artifact_from_a_run_that_did_not_finish_is_not_used(tmp_path):
    """A crashed run can leave a plausible-looking artifact behind. The registry
    is what decides, not the presence of a file."""
    for seed in SEEDS:
        write_eval(tmp_path, "nbpo", seed, 0.55)
    registry(tmp_path, [{"run_id": "eval-nbpo-s44", "phase": "eval", "method": "nbpo",
                         "seed": 44, "status": "failed",
                         "failure_reason": "OOM during decode"}])
    st = build(tmp_path, tmp_path / "out")
    assert st["per_row"]["nbpo"]["n_seeds"] == 2
    assert st["per_row"]["nbpo"]["complete"] is False
    assert 44 in st["per_row"]["nbpo"]["missing_seeds"]


def test_manuscript_update_requires_every_row(tmp_path):
    """One complete row is not licence to touch the manuscript."""
    for seed, w in zip(SEEDS, (0.54, 0.56, 0.55)):
        write_eval(tmp_path, "nbpo", seed, w)
    st = build(tmp_path, tmp_path / "out")
    assert st["rows_complete"] == 1
    assert st["manuscript_update_allowed"] is False


def test_row_order_is_fixed_and_does_not_depend_on_results(tmp_path):
    for seed, w in zip(SEEDS, (0.9, 0.9, 0.9)):
        write_eval(tmp_path, "game-ks", seed, w)      # make one row look best
    build(tmp_path, tmp_path / "out")
    tex = (tmp_path / "out" / "table1_generated.tex").read_text()
    order = [l.split(" &")[0] for l in tex.splitlines()
             if " & " in l and not l.startswith("Method")]
    assert order[0].startswith("Base")
    assert order[1] == "NBPO"
    assert order[2] == "Fixed-reference Nash"
    assert order[-1] == "RACO"
