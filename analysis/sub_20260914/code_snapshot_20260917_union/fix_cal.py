"""Add the two keys the materializer reads to the existing calibration file.

The pair files are already built and hashed, and their labels do not depend on
these keys -- they are metadata the materializer copies into the dataset's
provenance. So the file is amended in place rather than rebuilding 128,000 rows.
"""
import json
from pathlib import Path

p = Path("/work/uf4_20260910/dpo/safe_v1/calibration.json")
cal = json.loads(p.read_text())
before = sorted(cal)
cal.setdefault("weights_order", cal["objectives"])
if cal.get("tie_threshold_standardized") in (None, ""):
    report = json.loads(Path("/work/uf4_20260910/dpo/safe_v1/pairs/safe_uniform/"
                             "build_report_train.json").read_text())
    cal["tie_threshold_standardized"] = report["tie_threshold"]
p.write_text(json.dumps(cal, indent=2) + "\n")
print(json.dumps({"keys_before": before, "keys_after": sorted(cal),
                  "weights_order": cal["weights_order"],
                  "tie_threshold_standardized": cal["tie_threshold_standardized"]}, indent=1))
