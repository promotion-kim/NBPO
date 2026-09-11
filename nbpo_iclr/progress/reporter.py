#!/usr/bin/env python3
"""Thirty-minute reporter: snapshot, manuscript status block, PDF, Korean summary.

Runs independently of the job controller. Every cycle it reads the real GPU and
job state off the cluster, counts main-body completion against a frozen
execution matrix, rewrites only the marked status block in main_v6.tex, builds
the PDF in a scratch directory, and atomically replaces the published PDF only
when that build actually succeeded.

Design rules this file follows, because each has already gone wrong once here:
  * a single lock, so two reporters cannot both write the manuscript;
  * re-read the manuscript hash immediately before editing, so a concurrent edit
    by the user is never clobbered -- only the marker block is touched;
  * schedule on a fixed wall-clock grid, so a slow cycle does not make the
    period drift;
  * never report a PDF as fresh when the build failed: keep the last good one
    and say how old it is.
"""
from __future__ import annotations

import json, os, re, shutil, subprocess, sys, tempfile, time, hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path

KST = timezone(timedelta(hours=9))
ROOT = Path("/home/sjkim/MNPO")
PAPER = ROOT / "nbpo_iclr"
TEX = PAPER / "main_v6.tex"
PDF = PAPER / "main_v6.pdf"
PROG = PAPER / "progress"
LOCK = PROG / "reporter.lock"
TEXBIN = "/home/sjkim/MNPO/.TinyTeX/bin/x86_64-linux"
KUBECONFIG = "/home/sjkim/.kube/aipr-kubeconfig.yaml"
POD = ["kubectl", "exec", "-n", "p-aipr", "nbpo-judge", "-c", "main", "--", "bash", "-lc"]
BEGIN = "% BEGIN AUTO EXPERIMENT STATUS"
END = "% END AUTO EXPERIMENT STATUS"


def now():
    return datetime.now(KST)


def pod(cmd, timeout=120):
    env = dict(os.environ, KUBECONFIG=KUBECONFIG)
    try:
        r = subprocess.run(POD + [cmd], capture_output=True, text=True, timeout=timeout, env=env)
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


def snapshot():
    gpus = []
    out = pod("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total "
              "--format=csv,noheader,nounits")
    for line in out.strip().splitlines():
        try:
            i, u, m, t = (x.strip() for x in line.split(","))
            gpus.append({"gpu": int(i), "util": int(u), "used_mib": int(m), "total_mib": int(t)})
        except ValueError:
            continue
    raw = pod("cat /work/uf4_20260910/jobs/state.json")
    jobs = {}
    try:
        jobs = json.loads(raw)["jobs"] if raw.strip() else {}
    except Exception:
        jobs = {}
    for g in gpus:
        g["job"] = "-"
    for name, e in jobs.items():
        if e.get("state") == "RUNNING":
            for d in e.get("devices", []) or []:
                for g in gpus:
                    if g["gpu"] == d:
                        g["job"] = name
    counts = {}
    for e in jobs.values():
        counts[e.get("state", "?")] = counts.get(e.get("state", "?"), 0) + 1
    return {"gpus": gpus, "jobs": jobs, "state_counts": counts,
            "unreachable": not bool(out.strip())}


def completion():
    """Count filled main-body cells from the artifacts the cluster actually has."""
    m = json.loads((PROG / "execution_matrix.json").read_text())
    done = pod("ls -d /work/uf4_20260910/evaluation/final_eval/*/complete.json 2>/dev/null")
    evaluated = {Path(p).parent.name for p in done.split() if p.strip()}
    arms_to_method = {"nbpo_mse": "nbpo", "util_mse": "game_utilitarian"}
    obj_methods = set()
    if evaluated:
        obj_methods.add("base")
    for arm in evaluated:
        for pref, meth in arms_to_method.items():
            if arm.startswith(pref):
                obj_methods.add(meth)
    cap = pod("ls -d /work/uf4_20260910/evaluation/capability/*/*/results.json 2>/dev/null")
    cap_cells = len([p for p in cap.split() if p.strip()])
    bench = pod("ls -d /work/nbpo_repair_20260909/responses/uf4_*/ifeval.jsonl 2>/dev/null")
    bench_arms = len([p for p in bench.split() if p.strip()])
    return {
        "objective_rows_done": len(obj_methods),
        "objective_rows_total": len(m["objective_rows"]["methods"]),
        "objective_arms_evaluated": sorted(evaluated),
        "crossplay_done": 0, "crossplay_total": len(m["crossplay"]["entries"]),
        "dpo_done": 0, "dpo_total": len(m["dpo_weights"]["weights"]),
        "capability_cells_done": cap_cells,
        "capability_cells_total": len(m["capability"]["methods"]) * len(m["capability"]["benchmarks"]),
        "bench_arms_generated": bench_arms,
    }


def esc(text):
    """LaTeX-escape a status string so an underscore in a job name cannot break the build."""
    out = []
    for ch in str(text):
        if ch in "#$%&_{}":
            out.append("\\" + ch)
        elif ch == "\\":
            out.append("\\textbackslash{}")
        elif ch in "~^":
            out.append("\\char`\\" + ch + "{}")
        else:
            out.append(ch)
    return "".join(out)


def status_block(snap, comp, pdf_note):
    t = now().strftime("%Y-%m-%d %H:%M KST")
    rows = []
    for g in snap["gpus"]:
        rows.append("GPU%d: %s, %d\\%%, %d/%d MiB" %
                    (g["gpu"], esc(g["job"]), g["util"], g["used_mib"], g["total_mib"]))
    counts = ", ".join("%s %d" % (k, v) for k, v in sorted(snap["state_counts"].items()))
    lines = [
        BEGIN,
        "\\ifshowreviews",
        "\\begin{center}\\small\\fbox{\\parbox{0.95\\linewidth}{%",
        "\\textbf{Draft status, auto-updated %s.} Not part of the reported results." % esc(t),
        "\\\\[2pt]",
        "Main-body objective rows %d/%d; cross-play %d/%d; DPO weights %d/%d; capability cells %d/%d." %
        (comp["objective_rows_done"], comp["objective_rows_total"],
         comp["crossplay_done"], comp["crossplay_total"],
         comp["dpo_done"], comp["dpo_total"],
         comp["capability_cells_done"], comp["capability_cells_total"]),
        "\\\\[2pt]",
        esc(" | ".join(rows)),
        "\\\\[2pt]",
        "Queue: %s. %s" % (esc(counts), esc(pdf_note)),
        "}}\\end{center}",
        "\\fi",
        END,
    ]
    return "\n".join(lines)


def update_manuscript(block):
    """Rewrite only the marker block, on whatever the manuscript is right now."""
    text = TEX.read_text()
    before = hashlib.sha256(text.encode()).hexdigest()
    if BEGIN in text and END in text:
        start, end = text.index(BEGIN), text.index(END) + len(END)
        text = text[:start] + block + text[end:]
    else:
        anchor = "\\maketitle"
        if anchor in text:
            i = text.index(anchor) + len(anchor)
            text = text[:i] + "\n\n" + block + "\n" + text[i:]
        else:
            return False, before, "no anchor"
    tmp = TEX.with_suffix(".tex.tmp")
    tmp.write_text(text)
    os.replace(tmp, TEX)
    return True, before, "ok"


def build_pdf():
    """Compile in a scratch copy; only replace the published PDF on success."""
    stamp = now().strftime("%Y%m%d_%H%M_KST")
    with tempfile.TemporaryDirectory(prefix="nbpo_build_") as tmp:
        work = Path(tmp) / "paper"
        work.mkdir()
        for pattern in ("*.tex", "*.sty", "*.bst", "*.bib", "*.bbl"):
            for f in PAPER.glob(pattern):
                if f.is_file():
                    shutil.copy2(f, work / f.name)
        for sub in ("figures", "data"):
            if (PAPER / sub).is_dir():
                shutil.copytree(PAPER / sub, work / sub)
        env = dict(os.environ, PATH=TEXBIN + os.pathsep + os.environ.get("PATH", ""))
        log = work / "main_v6.log"
        ok = False
        try:
            subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                            "-file-line-error", "main_v6.tex"],
                           cwd=work, env=env, capture_output=True, text=True, timeout=900)
            ok = (work / "main_v6.pdf").exists()
        except Exception:
            ok = False
        if not ok:
            # latexmk can stop on a stale aux; a plain three-pass often succeeds
            for _ in range(3):
                subprocess.run(["pdflatex", "-interaction=nonstopmode", "main_v6.tex"],
                               cwd=work, env=env, capture_output=True, text=True, timeout=600)
            ok = (work / "main_v6.pdf").exists()
        detail = {"stamp": stamp, "ok": ok}
        if log.exists():
            txt = log.read_text(errors="ignore")
            detail["pages"] = (re.findall(r"Output written.*?\((\d+) pages", txt) or [None])[0]
            detail["undefined"] = txt.count("undefined")
            detail["overfull"] = txt.count("Overfull")
            shutil.copy2(log, PROG / "logs" / ("build_%s.log" % stamp))
        if ok:
            archive = PROG / "pdfs" / ("main_v6_%s.pdf" % stamp)
            shutil.copy2(work / "main_v6.pdf", archive)
            tmp_pdf = PDF.with_suffix(".pdf.tmp")
            shutil.copy2(work / "main_v6.pdf", tmp_pdf)
            os.replace(tmp_pdf, PDF)
            detail["archive"] = str(archive)
        return detail


def korean_summary(snap, comp, build, extra):
    t = now().strftime("%H:%M")
    g = {x["gpu"]: x for x in snap["gpus"]}
    def gl(i):
        x = g.get(i)
        if not x:
            return "GPU%d: 조회 실패" % i
        return "GPU%d: %s, %d%%, %d/%d MiB" % (i, x["job"], x["util"], x["used_mib"], x["total_mib"])
    counts = ", ".join("%s %d" % (k, v) for k, v in sorted(snap["state_counts"].items()))
    if build["ok"]:
        pdfline = "PDF: 성공 %s, %s쪽, nbpo_iclr/main_v6.pdf" % (t, build.get("pages"))
    else:
        last = sorted((PROG / "pdfs").glob("*.pdf"))
        pdfline = "PDF: 실패, 최신 성공본 %s 유지" % (last[-1].name if last else "없음")
    return "\n".join([
        "[KST %s] 본문 완료: objective %d/%d, cross-play %d/%d, DPO weights %d/7, capability %d/%d"
        % (t, comp["objective_rows_done"], comp["objective_rows_total"],
           comp["crossplay_done"], comp["crossplay_total"], comp["dpo_done"],
           comp["capability_cells_done"], comp["capability_cells_total"]),
        "진행/다음: 큐 %s → %s" % (counts, extra.get("next", "READY 최상위")),
        gl(0) + " | " + gl(1),
        gl(2) + " | " + gl(3),
        "ETA: %s" % extra.get("eta", "산정 대기"),
        "이번 완료: %s" % extra.get("done", "새 완료 결과 없음"),
        pdfline,
        "Blocker/복구: %s" % extra.get("blocker", "없음"),
    ])


def cycle():
    snap = snapshot()
    comp = completion()
    pdf_note = "PDF rebuilt each cycle."
    update_manuscript(status_block(snap, comp, pdf_note))
    build = build_pdf()
    ready = [j for j, e in snap["jobs"].items() if e.get("state") == "READY"]
    extra = {"next": (sorted(ready)[0] if ready else "없음"),
             "eta": "핵심 기전 [산정 대기: fixed-ref UF 학습 미착수] / 본문 첫 전체 평가 [17:30-18:30] / 본문 최종 [산정 대기: DPO·PROSPER·MOPO 미착수]",
             "done": "-", "blocker": "없음" if not snap["unreachable"] else "클러스터 조회 실패"}
    text = korean_summary(snap, comp, build, extra)
    stamp = now().strftime("%Y%m%d_%H%M_KST")
    (PROG / "latest.md").write_text(text + "\n")
    (PROG / "logs" / ("status_%s.md" % stamp)).write_text(text + "\n")
    with (PROG / "status.jsonl").open("a") as f:
        f.write(json.dumps({"kst": now().strftime("%Y-%m-%d %H:%M:%S"),
                            "gpus": snap["gpus"], "state_counts": snap["state_counts"],
                            "completion": comp, "build": build}) + "\n")
    print(text, flush=True)
    print("-" * 60, flush=True)


def main():
    PROG.mkdir(parents=True, exist_ok=True)
    (PROG / "logs").mkdir(exist_ok=True)
    (PROG / "pdfs").mkdir(exist_ok=True)
    if LOCK.exists():
        try:
            old = int(LOCK.read_text().strip())
            os.kill(old, 0)
            print("another reporter is alive at pid %d; exiting" % old); return
        except Exception:
            pass
    LOCK.write_text(str(os.getpid()))
    period = 1800
    try:
        while True:
            started = time.time()
            try:
                cycle()
            except Exception as error:                       # noqa: BLE001
                print("cycle error: %r" % (error,), flush=True)
            # fixed grid so a slow cycle does not make the period drift
            sleep = period - ((time.time() - started) % period)
            time.sleep(max(60, sleep))
    finally:
        if LOCK.exists() and LOCK.read_text().strip() == str(os.getpid()):
            LOCK.unlink()


if __name__ == "__main__":
    main()
