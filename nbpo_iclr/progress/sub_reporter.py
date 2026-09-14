#!/usr/bin/env python3
"""Submission-campaign reporter: status only, never an edit to the manuscript.

The 2026-09-14 package protects the manuscript by whole-file sha256, so the old
reporter's habit of rewriting its AUTO regions every cycle would break the
contract on the first tick. This reporter reads state, writes
progress/latest.md, and touches the paper only through the declared results
path:

    protection.status() -> render_results.py -> inline_inputs.py
                        -> pdflatex -> protection.status()

main_v6.tex is self-contained by request, so the rendered result block reaches
it through inline_inputs.py, and protection.status() is what judges the
contract: eleven files byte-identical, and main_v6.tex accepted only while it
collapses back to the declared bytes.

If that verdict fails at either end the build is skipped and the last good PDF
is kept, because a protected-file mismatch is a reason to stop writing PDFs,
not to overwrite one. latexmk is unavailable on this host (perl Time::HiRes is
missing), so the build is pdflatex + bibtex + two passes.

Counts come from artifacts, never from intent: a measured cell is one with a
record in templates/results.json, and a finished audit is one with a
complete.json on the pod.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
PROG = PAPER / "progress"
LOCK = PROG / "sub_reporter.lock"
TEXBIN = "/home/sjkim/MNPO/.TinyTeX/bin/x86_64-linux"
KUBECONFIG = "/home/sjkim/.kube/aipr-kubeconfig.yaml"
POD = ["kubectl", "exec", "-n", "p-aipr", "nbpo-judge", "-c", "main", "--", "bash", "-lc"]
SUB = "/work/sub_20260914"
UF = "/work/uf4_20260910"
AUDITS = 2          # Safe and Wild
FACTOR_CELLS = 4
METHODS = 7         # the declared policy panel, one seed each in this campaign
EVAL_ARMS = 8       # seven trained arms plus the base fresh draw


def pod(cmd, timeout=120):
    env = dict(os.environ, KUBECONFIG=KUBECONFIG)
    try:
        r = subprocess.run(POD + [cmd], capture_output=True, text=True,
                           timeout=timeout, env=env)
        return r.stdout
    except Exception:                                            # noqa: BLE001
        return ""


def gpus():
    out = pod("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total "
              "--format=csv,noheader,nounits")
    rows = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            rows[int(parts[0])] = {"util": int(parts[1]), "used": int(parts[2]),
                                   "total": int(parts[3]), "job": "-"}
        except ValueError:
            continue
    return rows


def queue_state():
    raw = pod("cd %s && python3 - <<'PY'\n"
              "import json, glob\n"
              "st = json.load(open('jobs/state.json'))['jobs']\n"
              "specs = {json.loads(open(p).read())['job_id']: json.loads(open(p).read())\n"
              "         for p in glob.glob('jobs/queue/*.json')}\n"
              "live = {j: v for j, v in st.items() if j in specs}\n"
              "counts = {}\n"
              "for v in live.values():\n"
              "    counts[v['state']] = counts.get(v['state'], 0) + 1\n"
              "running = {j: v.get('devices') for j, v in live.items() if v['state'] == 'RUNNING'}\n"
              "ready = sorted((specs[j]['priority'], j) for j, v in live.items() if v['state'] == 'READY')\n"
              "stale = sorted(j for j, v in st.items() if j not in specs and v['state'] in ('FAILED', 'BLOCKED', 'READY', 'PENDING'))\n"
              "failed = sorted(j for j, v in live.items() if v['state'] == 'FAILED')\n"
              "print(json.dumps({'counts': counts, 'running': running, 'ready': ready,\n"
              "                  'stale_rows': len(stale), 'live_failed': failed}))\n"
              "PY" % UF)
    for line in reversed(raw.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {"counts": {}, "running": {}, "ready": [], "stale_rows": 0, "live_failed": []}


def campaign():
    raw = pod("python3 - <<'PY'\n"
              "import json, glob, os\n"
              "S = '%s'\n"
              "out = {'judged': [], 'pools': [], 'factor_cells': 0, 'factor_summary': None}\n"
              "for f in sorted(glob.glob(S + '/judgments/*/complete.json')):\n"
              "    d = json.load(open(f))\n"
              "    out['judged'].append({'tag': os.path.basename(os.path.dirname(f)),\n"
              "                          'prompts': d['n_prompts'], 'verdicts': d['n_judgments'],\n"
              "                          'rate': d['judgments_per_second'], 'parse': d['valid_fraction']})\n"
              "for f in sorted(glob.glob('%s/pools/safe*/shard*/complete*.json')):\n"
              "    out['pools'].append(f)\n"
              "p = S + '/factorial/summary.json'\n"
              "if os.path.exists(p):\n"
              "    d = json.load(open(p))\n"
              "    out['factor_cells'] = sum(1 for v in d['cells'].values() if v['instances_usable'])\n"
              "    out['factor_summary'] = {c: {'C': v['C']['mean'], 'D': v['D']['mean'],\n"
              "                                 'gamma': v['gamma_star']['mean'],\n"
              "                                 'dM': v['Delta_M']['mean'], 'TV': v['target_TV']['mean'],\n"
              "                                 'n': v['instances_usable']}\n"
              "                             for c, v in d['cells'].items()}\n"
              "print(json.dumps(out))\n"
              "PY" % (SUB, UF))
    for line in reversed(raw.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {"judged": [], "pools": [], "factor_cells": 0, "factor_summary": None}


def protected():
    """The whole contract in one verdict, see progress/protection.py."""
    sys.path.insert(0, str(PROG))
    from protection import status                                 # noqa: PLC0415
    return status()


def measured_cells():
    doc = json.loads((PAPER / "templates/results.json").read_text())
    cells = doc["cells"]
    return sum(1 for v in cells.values() if v is not None), len(cells)


def build():
    env = dict(os.environ, PATH=TEXBIN + os.pathsep + os.environ.get("PATH", ""))
    logs = PROG / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in ("main_v6", "main_clean"):
        try:
            subprocess.run(["pdflatex", "-interaction=nonstopmode", name + ".tex"],
                           cwd=PAPER, env=env, capture_output=True, text=True, timeout=900)
            subprocess.run(["bibtex", name], cwd=PAPER, env=env,
                           capture_output=True, text=True, timeout=300)
            for _ in range(2):
                r = subprocess.run(["pdflatex", "-interaction=nonstopmode", name + ".tex"],
                                   cwd=PAPER, env=env, capture_output=True, text=True,
                                   timeout=900)
            (logs / (name + ".log")).write_text(r.stdout[-40000:])
            pdf = PAPER / (name + ".pdf")
            pages = None
            if pdf.exists():
                info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True)
                m = re.search(r"^Pages:\s+(\d+)", info.stdout, re.M)
                pages = m.group(1) if m else None
            errors = len(re.findall(r"^! ", r.stdout, re.M))
            results[name] = {"ok": pdf.exists() and errors == 0, "pages": pages,
                             "errors": errors}
        except Exception as error:                               # noqa: BLE001
            results[name] = {"ok": False, "pages": None, "errors": -1,
                             "why": repr(error)[:120]}
    return results


def cycle():
    kst = time.strftime("%H:%M", time.localtime(time.time() + 0))
    ok_before, msg_before = protected()
    render = subprocess.run([sys.executable, "render_results.py"], cwd=PAPER,
                            capture_output=True, text=True)
    # carry the rendered result block into the self-contained manuscript
    subprocess.run([sys.executable, "progress/inline_inputs.py"], cwd=PAPER,
                   capture_output=True, text=True)
    ok_after, msg_after = protected()
    builds = {}
    if ok_before and ok_after:
        builds = build()
    g, q, camp = gpus(), queue_state(), campaign()
    for job, devices in (q.get("running") or {}).items():
        for d in devices or []:
            if d in g:
                g[d]["job"] = job
    measured, declared = measured_cells()

    audits_done = sum(1 for j in camp["judged"]
                      if j["tag"].startswith("safe_rest") or j["tag"].startswith("safe_screen"))
    safe_done = any(j["tag"].startswith("safe_rest") for j in camp["judged"])
    verdicts = sum(j["verdicts"] for j in camp["judged"])
    rates = [j["rate"] for j in camp["judged"]] or [0]

    def gl(i):
        x = g.get(i)
        if not x:
            return "GPU%d: 조회 실패" % i
        return "GPU%d: %s, %d%%, %d/%d MiB" % (i, x["job"], x["util"], x["used"], x["total"])

    idle = [i for i, x in g.items() if x["job"] == "-" and x["util"] < 5]
    ready = q.get("ready") or []
    blocker = "없음"
    if idle and ready:
        blocker = "GPU %s 유휴, READY 최상위 %s 대기" % (idle, ready[0][1])
    elif idle and not ready:
        blocker = "GPU %s 유휴, dispatch 가능한 READY 없음" % idle
    if not ok_before or not ok_after:
        blocker = "보호 검증 실패(%s) → PDF 미갱신" % (msg_after or msg_before)
    if q.get("live_failed"):
        blocker = "실패 job %s" % ", ".join(q["live_failed"][:3])

    pdfline = "PDF: " + ", ".join(
        "%s %s쪽%s" % (n, v.get("pages"), "" if v.get("ok") else " (실패)")
        for n, v in builds.items()) if builds else "PDF: 보호 검증 실패로 건너뜀"

    lines = [
        "[KST %s] 새 panel PKU-SafeRLHF/help-safe · 보호 검증: %s"
        % (kst, "PASS" if (ok_before and ok_after) else "FAIL " + (msg_after or msg_before)),
        "완료: audit %d/%d, finite cells %d/%d, methods×seeds %d/%d, final eval %d/%d, template cell %d/%d"
        % (int(safe_done), AUDITS, camp["factor_cells"], FACTOR_CELLS,
           0, METHODS, 0, EVAL_ARMS, measured, declared),
        "진행: 판정 누적 %d verdict, 실측 %.1f verdict/s/GPU · 큐 %s · 다음 READY %s"
        % (verdicts, max(rates), q.get("counts", {}),
           ready[0][1] if ready else "없음"),
        gl(0) + " | " + gl(1),
        gl(2) + " | " + gl(3),
        "ETA: 별도 보고 참조(실측 기반)",
        "이번 결과: %s" % ("측정 cell %d개" % measured if measured else "새 template cell 없음"),
        pdfline,
        "Blocker: %s" % blocker,
    ]
    text = "\n".join(lines) + "\n"
    tmp = PROG / "latest.md.tmp"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(PROG / "latest.md")
    print(text, flush=True)
    (PROG / "sub_status.json").write_text(json.dumps(
        {"kst": kst, "protected_ok": ok_before and ok_after, "builds": builds,
         "queue": q, "campaign": camp, "measured_cells": measured,
         "render": (render.stdout or render.stderr).strip()[:200]},
        indent=1, default=str) + "\n")


def main():
    if LOCK.exists():
        try:
            os.kill(int(LOCK.read_text().strip()), 0)
            print("another sub_reporter is alive at pid %s" % LOCK.read_text().strip())
            return
        except Exception:                                        # noqa: BLE001
            pass
    LOCK.write_text(str(os.getpid()))
    once = "--once" in sys.argv
    period = 1800
    try:
        while True:
            started = time.time()
            try:
                cycle()
            except Exception as error:                           # noqa: BLE001
                print("cycle error: %r" % (error,), flush=True)
            if once:
                return
            time.sleep(max(60, period - ((time.time() - started) % period)))
    finally:
        if LOCK.exists() and LOCK.read_text().strip() == str(os.getpid()):
            LOCK.unlink()


if __name__ == "__main__":
    main()
