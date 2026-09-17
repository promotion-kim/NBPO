"""Fix two defects in the diagnostics manifest.

A finished job whose declared artifact path was wrong showed as NOT_STARTED,
because the status chain had no branch for the controller's own DONE state and
fell through to the default. Both pilot rows hit that: their real outputs are
the exported checkpoint and the shared pilot table, not the per-arm
complete.json the row invented.
"""
from pathlib import Path

PATH = Path("/work/uf4_20260910/code/diag_manifest.py")
src = PATH.read_text()

OLD = '''    ("projection_sampled", "tab:projection-ablation", "sampled-pair regression", 42,
     "uf4_diag_proj_sampled", DIAG / "projection/sampled/complete.json", 0.8),
    ("projection_all", "tab:projection-ablation", "all-candidate regression", 42,
     "uf4_diag_proj_all", DIAG / "projection/all/complete.json", 0.8),'''
NEW = '''    ("projection_sampled", "tab:projection-ablation", "sampled-pair regression", 42,
     "uf4_diag_proj_sampled", ROOT / "arms/diag_proj_sampled_s42/config.json", 1.07),
    ("projection_all", "tab:projection-ablation", "all-candidate regression", 42,
     "uf4_diag_proj_all", ROOT / "arms/diag_proj_all_s42/config.json", 0.93),
    ("projection_fit", "tab:projection-ablation", "pilot fit and pool metrics", 42,
     "uf4_diag_pilot_fit", DIAG / "projection/pilot_table.json", 0.1),'''
assert src.count(OLD) == 1
src = src.replace(OLD, NEW, 1)

OLD2 = '''        if artifact.exists():
            status = "DONE"
        elif job and entry.get("state") == "RUNNING":'''
NEW2 = '''        if artifact.exists():
            status = "DONE"
        elif job and entry.get("state") == "DONE":
            # the controller finished it; the row's declared artifact path is the
            # thing that is wrong, and saying NOT_STARTED here would be a lie
            status = "DONE_ARTIFACT_PATH_MISMATCH"
        elif job and entry.get("state") == "RUNNING":'''
assert src.count(OLD2) == 1
src = src.replace(OLD2, NEW2, 1)
PATH.write_text(src)
import ast
ast.parse(src)
print("manifest status chain and pilot artifact paths fixed")
