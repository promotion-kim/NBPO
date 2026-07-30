#!/usr/bin/env python3
"""Atomic shared-filesystem scheduler worker for one B200."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path


ARMS = ["ronpo_os", "ronpo_topmass", "inpo_avg", "sppo_avg", "simpo", "ipo", "dpo", "ht_mnpo_harmless", "ht_mnpo_helpfulness"]


def gate(root: Path, stage: int, arm: str) -> Path:
    base = root / ("stage12" if stage <= 2 else f"stage{stage}")
    return base / f"stage{stage}_stability_p8_locked_panel/gates/{arm}.json"


def gate_passed(path: Path) -> bool:
    try:
        d = json.loads(path.read_text(encoding="utf-8")); return d.get("passed") is True and d.get("status") == "passed"
    except (FileNotFoundError, json.JSONDecodeError): return False


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--project", type=Path, required=True); p.add_argument("--venv", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--base", required=True); p.add_argument("--gpu", type=int, required=True)
    a = p.parse_args()
    sched = a.root / "stage4/scheduler"; (sched / "claims").mkdir(parents=True, exist_ok=True); (sched / "logs").mkdir(parents=True, exist_ok=True)
    worker_id = f"{socket.gethostname()}-g{a.gpu}"; worker_log = sched / f"worker_{worker_id}.log"
    tasks = [(stage, arm) for stage in range(1, 5) for arm in ARMS]
    while True:
        claimed = False
        for stage, arm in tasks:
            name = f"stage{stage}__{arm}"; done = sched / f"{name}.DONE.json"; failed = sched / f"{name}.FAILED.json"
            if done.exists() or failed.exists(): continue
            if stage > 1 and not gate_passed(gate(a.root, stage - 1, arm)): continue
            claim = sched / "claims" / name
            try: claim.mkdir()
            except FileExistsError: continue
            (claim / "owner.json").write_text(json.dumps({"worker": worker_id, "pid": os.getpid(), "claimed": time.time()}, indent=2) + "\n", encoding="utf-8")
            command = [str(a.venv / "bin/python"), str(a.project / "analysis/p13_saferlhf_seed44_stage4_20260718/run_stage_task.py"),
                       "--project", str(a.project), "--venv", str(a.venv), "--stage12", str(a.root / "stage12"),
                       "--stage3", str(a.root / "stage3"), "--stage4", str(a.root / "stage4"), "--base", a.base,
                       "--stage", str(stage), "--arm", arm, "--gpu", str(a.gpu), "--log", str(sched / "logs" / f"{name}_{worker_id}.log")]
            started = time.time()
            result = subprocess.run(command)
            payload = {"worker": worker_id, "stage": stage, "arm": arm, "returncode": result.returncode,
                       "started": started, "finished": time.time(), "elapsed_seconds": time.time() - started}
            (done if result.returncode == 0 else failed).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            with worker_log.open("a", encoding="utf-8") as h: h.write(json.dumps(payload) + "\n")
            claimed = True; break
        if claimed: continue
        terminal = sum((sched / f"stage{s}__{arm}.DONE.json").exists() or (sched / f"stage{s}__{arm}.FAILED.json").exists() for s, arm in tasks)
        active = [p for p in (sched / "claims").iterdir() if not (sched / f"{p.name}.DONE.json").exists() and not (sched / f"{p.name}.FAILED.json").exists()]
        if terminal == len(tasks) or (not active and any((sched / f"stage{s}__{arm}.FAILED.json").exists() for s, arm in tasks)):
            break
        time.sleep(20)


if __name__ == "__main__": main()
